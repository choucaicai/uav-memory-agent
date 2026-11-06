"""Continuous, metre-coordinate rotary embedding for cross-attention Q/K."""
import torch


def apply_rope3d(tensor, world_xyz, base=10000.0):
    # tensor [B,H,N,D], coordinates [B,N,3]. V is never passed here.
    if world_xyz.shape != (tensor.shape[0], tensor.shape[2], 3):
        raise ValueError(f"RoPE coordinate/token mismatch: {world_xyz.shape}, {tensor.shape}")
    axis_dim = 2 * (tensor.shape[-1] // 6)
    if axis_dim == 0:
        raise ValueError("3D RoPE requires at least six dimensions per head")
    with torch.autocast(device_type=tensor.device.type, enabled=False):
        frequencies = base ** (-torch.arange(0, axis_dim, 2, device=tensor.device, dtype=torch.float32) / axis_dim)
        chunks = []
        for axis in range(3):
            part = tensor[..., axis * axis_dim:(axis + 1) * axis_dim]
            phase = world_xyz[..., axis].float()[:, None, :, None] * frequencies
            cos, sin = phase.cos().to(tensor.dtype), phase.sin().to(tensor.dtype)
            even, odd = part[..., 0::2], part[..., 1::2]
            chunks.append(torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2))
    return torch.cat(chunks + [tensor[..., 3 * axis_dim:]], dim=-1)
