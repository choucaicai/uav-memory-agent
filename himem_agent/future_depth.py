"""Causal depth-query embeddings, one-upsample head, and masked log-depth loss."""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class FutureDepthHead(nn.Module):
    def __init__(self, hidden_size, channels=256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, channels)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.SiLU(),
        )
        self.output = nn.Conv2d(64, 1, 1)
        self.reset_output()

    def reset_output(self):
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.constant_(self.output.bias, math.log(20.0))

    def forward(self, hidden):
        if hidden.ndim != 3 or hidden.shape[1] != 64:
            raise ValueError('Depth head expects [B, 64, hidden_size]')
        x = self.proj(self.norm(hidden)).transpose(1, 2)
        x = x.reshape(hidden.shape[0], -1, 8, 8)
        x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        return self.output(self.refine(x)).float()


def grid_position(hidden_size):
    """Fixed row/column sine-cosine anchors; independent of all target geometry."""
    if hidden_size % 4:
        raise ValueError('Depth query hidden size must be divisible by four')
    freq = 10000.0 ** (-torch.arange(hidden_size // 4).float() / (hidden_size // 4))
    row, col = torch.meshgrid(torch.arange(8), torch.arange(8), indexing='ij')
    row = row.flatten().float()[:, None] * freq
    col = col.flatten().float()[:, None] * freq
    return .02 * torch.cat((row.sin(), row.cos(), col.sin(), col.cos()), -1)[None]


class FutureDepthAux(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, 64, hidden_size))
        self.readout = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.register_buffer('grid_position', grid_position(hidden_size))
        self.head = FutureDepthHead(hidden_size)
        self.reset_queries()

    def reset_queries(self):
        nn.init.normal_(self.queries, std=.02)
        nn.init.normal_(self.readout, std=.02)

    def query_embeddings(self):
        return self.queries + self.grid_position.to(self.queries.dtype)


def depth_loss_terms(pred_log, target, mask, valid, *, min_depth=.1, max_depth=100.,
                     near_distance=10., near_weight=3., huber_delta=1.):
    """Return differentiable numerators and target-only denominators.

    Sum terms across time/ranks before dividing. An all-invalid batch still
    connects the entire head to the graph with zero gradients.
    """
    if target.shape != pred_log.shape or mask.shape != pred_log.shape:
        raise ValueError('Depth prediction, target and mask must have shape [B, 1, 16, 16]')
    if valid.shape != (pred_log.shape[0],):
        raise ValueError('future_valid must have shape [B]')
    pred_log, target = pred_log.float(), target.float()
    m = (mask.bool() & valid.bool()[:, None, None, None] & torch.isfinite(target)
         & (target >= min_depth) & (target <= max_depth))
    safe_target = torch.where(m, target, torch.ones_like(target))
    z = safe_target.log()
    weight = torch.where(safe_target < near_distance, near_weight, 1.) * m.float()
    reg = F.huber_loss(pred_log, z, reduction='none', delta=huber_delta)
    reg_sum, reg_count = (reg * weight).sum(), weight.sum()
    edge_sum = pred_log.sum() * 0.
    edge_count = pred_log.new_zeros(())
    for dim in (-1, -2):
        edge_mask = m.narrow(dim, 1, m.shape[dim] - 1) & m.narrow(dim, 0, m.shape[dim] - 1)
        error = (torch.diff(pred_log, dim=dim) - torch.diff(z, dim=dim)).abs()
        edge_sum = edge_sum + (error * edge_mask).sum()
        edge_count = edge_count + edge_mask.sum()
    return reg_sum, reg_count, edge_sum, edge_count, m.flatten(1).any(1).sum()


@dataclass
class PackedDepthSequence:
    embeddings: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    depth_positions: torch.Tensor


def prompt_masks(attention_mask, labels=None, prompt_lengths=None):
    """Identify the prompt before the first answer label, never by token value."""
    valid = attention_mask.bool()
    prompt = torch.zeros_like(valid)
    lengths = []
    for i in range(len(valid)):
        positions = valid[i].nonzero(as_tuple=True)[0]
        n = len(positions)
        supervised = None if labels is None else labels[i, positions].ne(-100).nonzero(as_tuple=True)[0]
        inferred = n if supervised is None or not len(supervised) else int(supervised[0])
        cut = inferred if prompt_lengths is None else int(prompt_lengths[i])
        if not 1 <= cut <= n:
            raise ValueError(f'Invalid prompt length {cut} for {n} valid text tokens')
        if supervised is not None and len(supervised) and cut != inferred:
            raise ValueError('prompt_lengths must end at the first supervised action token')
        if labels is not None and (supervised is None or not len(supervised)):
            raise ValueError('Supervised forward requires action labels after the prompt')
        prompt[i, positions[:cut]] = True
        lengths.append(cut)
    return prompt, lengths


def pack_depth_sequence(text, visual, attention_mask, labels, prompt_lengths, aux):
    """[BOS, visual, prompt, 64 queries, readout, action/EOS], right padded."""
    _, cuts = prompt_masks(attention_mask, labels, prompt_lengths)
    rows, targets, offsets = [], [], []
    for i, cut in enumerate(cuts):
        keep = attention_mask[i].bool()
        words = text[i, keep]
        labs = (torch.full((len(words),), -100, dtype=torch.long, device=text.device)
                if labels is None else labels[i, keep])
        prefix = torch.cat((words[:1], visual[i], words[1:cut]))
        queries = aux.query_embeddings()[0].to(text.dtype)
        readout = aux.readout[0].to(text.dtype)
        rows.append(torch.cat((prefix, queries, readout, words[cut:])))
        targets.append(torch.cat((labs.new_full((len(prefix) + 65,), -100), labs[cut:])))
        offsets.append(len(prefix))
    embeddings = nn.utils.rnn.pad_sequence(rows, batch_first=True)
    new_labels = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
    mask = torch.arange(embeddings.shape[1], device=text.device)[None] < torch.tensor(
        [len(row) for row in rows], device=text.device)[:, None]
    positions = mask.long().cumsum(-1).sub(1).clamp_min(0)
    query_positions = torch.tensor(offsets, device=text.device)[:, None] + torch.arange(64, device=text.device)
    return PackedDepthSequence(embeddings, new_labels, mask, positions, query_positions)
