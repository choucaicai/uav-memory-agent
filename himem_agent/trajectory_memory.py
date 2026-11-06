"""World-keyed, differentiable fine/coarse memory. Runtime states are not weights."""
from dataclasses import dataclass, replace
import torch
from torch import nn


@dataclass
class CellTable:
    keys: torch.Tensor
    hidden: torch.Tensor
    xyz_sum: torch.Tensor
    count: torch.Tensor
    visited: torch.Tensor

    @property
    def world_xyz(self):
        return self.xyz_sum / self.count.clamp_min(1)[:, None]

    def detach(self):
        return replace(self, hidden=self.hidden.detach())


@dataclass
class EpisodeMemory:
    fine: CellTable
    coarse: CellTable
    poses: torch.Tensor
    step_id: int = -1

    def detach(self):
        return replace(self, fine=self.fine.detach(), coarse=self.coarse.detach())


def empty_table(features):
    return CellTable(
        torch.empty((0, 3), dtype=torch.long, device=features.device),
        features.new_empty((0, features.shape[-1])),
        torch.empty((0, 3), dtype=torch.float32, device=features.device),
        torch.empty(0, dtype=torch.long, device=features.device),
        torch.empty(0, dtype=torch.long, device=features.device),
    )


def covered(points, poses, radius):
    # Bound the temporary distance matrix for long episodes.
    result = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    for start in range(0, len(poses), 64):
        delta = points[:, None] - poses[None, start:start + 64]
        result = result | (delta.square().sum(-1) <= radius * radius).any(-1)
    return result


def pool_cells(features, xyz, voxel_size, point_visited):
    keys, inverse = torch.unique(torch.round(xyz / voxel_size).long(), dim=0, return_inverse=True)
    n = len(keys)
    count = torch.bincount(inverse, minlength=n)
    # FP32 accumulation avoids loss of precision for geometry and large cells.
    sums = torch.zeros(n, features.shape[-1], dtype=torch.float32, device=features.device)
    sums = sums.index_add(0, inverse, features.float())
    hidden = (sums / count.clamp_min(1)[:, None]).to(features.dtype)
    xyz_sum = xyz.new_zeros(n, 3).index_add(0, inverse, xyz)
    visited_count = torch.zeros(n, dtype=torch.long, device=xyz.device).index_add(
        0, inverse, point_visited.long())
    return CellTable(keys, hidden, xyz_sum, count, (visited_count > 0).long())


class TrajectoryMemory(nn.Module):
    def __init__(self, dim, fine_voxel_size=5.0, coarse_voxel_size=25.0, visit_radius=20.0,
                 use_fine=True, use_coarse=True, use_visited=True):
        super().__init__()
        if not (0 < fine_voxel_size < coarse_voxel_size and visit_radius > 0):
            raise ValueError("Require 0 < fine voxel < coarse voxel and positive visit radius")
        if not (use_fine or use_coarse):
            raise ValueError("At least one of fine/coarse memory must be enabled")
        # Component ablation switches; defaults reproduce the full model.
        self.use_fine = bool(use_fine)
        self.use_coarse = bool(use_coarse)
        self.use_visited = bool(use_visited)
        self.fine_voxel_size = fine_voxel_size
        self.coarse_voxel_size = coarse_voxel_size
        self.visit_radius = visit_radius
        # Kept only so older checkpoints load strictly; cell updates no longer use it.
        self.updater = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.updater.requires_grad_(False)
        self.visited_embedding = nn.Embedding(3, dim)
        # Recurrent cell states are fed back on every observation.  Keep normal
        # feature magnitudes unchanged while preventing one episode from
        # amplifying a BF16 state until the next update's gradients overflow.
        self.hidden_clip = 4096.0

    def merge(self, old, current):
        keys, inverse = torch.unique(torch.cat([old.keys, current.keys]), dim=0, return_inverse=True)
        old_slots, cur_slots = inverse[:len(old.keys)], inverse[len(old.keys):]
        n, dim = len(keys), old.hidden.shape[-1]
        old_aligned = old.hidden.new_zeros(n, dim).index_copy(0, old_slots, old.hidden)
        cur_aligned = current.hidden.new_zeros(n, dim).index_copy(0, cur_slots, current.hidden)
        has_old = torch.zeros(n, dtype=torch.bool, device=keys.device)
        has_cur = torch.zeros_like(has_old)
        has_old[old_slots], has_cur[cur_slots] = True, True
        # Non-parametric cell update: old-only cells keep their history,
        # new-only cells take the current visual feature, and overlapping cells
        # use an equal average of the two cell summaries.  count is only
        # geometry/visit bookkeeping and intentionally does not weight hidden.
        has_old_f = has_old[:, None].to(torch.float32)
        has_cur_f = has_cur[:, None].to(torch.float32)
        hidden = (
            old_aligned.float() * has_old_f
            + cur_aligned.float() * has_cur_f
        ) / (has_old_f + has_cur_f).clamp_min(1.0)
        hidden = hidden.to(current.hidden.dtype)
        xyz_sum = old.xyz_sum.new_zeros(n, 3).index_add(0, old_slots, old.xyz_sum)
        xyz_sum = xyz_sum.index_add(0, cur_slots, current.xyz_sum)
        count = old.count.new_zeros(n).index_add(0, old_slots, old.count)
        count = count.index_add(0, cur_slots, current.count)
        visits = old.visited.new_zeros(n).index_add(0, old_slots, old.visited)
        visits = visits.index_add(0, cur_slots, current.visited)
        return CellTable(keys, hidden, xyz_sum, count, (visits > 0).long())

    def forward(self, state, features, world_xyz, agent_position, step_id):
        xyz = world_xyz.detach().float()
        pose = agent_position.detach().float().reshape(1, 3)
        if state is None:
            state = EpisodeMemory(empty_table(features), empty_table(features), pose[:0])
        if step_id == state.step_id:
            # Generation can revisit a prefill; never count the same observation twice.
            return state
        if step_id != state.step_id + 1:
            raise ValueError(f"Nonconsecutive memory step: previous={state.step_id}, current={step_id}")
        poses = torch.cat([state.poses, pose])
        near = (xyz - pose).square().sum(-1) <= self.visit_radius ** 2
        visits = covered(xyz, poses, self.visit_radius)
        # A disabled level keeps an empty table, so it contributes no memory tokens.
        fine, coarse = state.fine, state.coarse
        if self.use_fine:
            fine_obs = pool_cells(features[near], xyz[near], self.fine_voxel_size, visits[near])
            fine = self.merge(state.fine, fine_obs)
        if self.use_coarse:
            coarse_obs = pool_cells(features, xyz, self.coarse_voxel_size, visits)
            coarse = self.merge(state.coarse, coarse_obs)
        for table in (fine, coarse):
            table.visited = (table.visited.bool() | covered(table.world_xyz, pose, self.visit_radius)).long()
        return EpisodeMemory(fine, coarse, poses, step_id)

    def read(self, state):
        hidden = torch.cat([state.fine.hidden, state.coarse.hidden])
        visited = torch.cat([state.fine.visited, state.coarse.visited])
        visited_tokens = self.visited_embedding(visited)
        if not self.use_visited:
            # Keep the parameter in the graph (DDP has find_unused_parameters=False) with zero effect.
            visited_tokens = visited_tokens * 0.0
        tokens = hidden + visited_tokens
        xyz = torch.cat([state.fine.world_xyz, state.coarse.world_xyz])
        fine_mask = torch.arange(len(tokens), device=tokens.device) < len(state.fine.keys)
        coarse_mask = ~fine_mask
        if not (self.use_fine and self.use_coarse):
            # Single-level ablation: every query group reads the remaining level.
            fine_mask = torch.ones_like(fine_mask)
            coarse_mask = fine_mask
        return tokens, xyz, fine_mask, coarse_mask
