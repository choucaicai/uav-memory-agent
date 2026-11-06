"""Depth labels; the target selection never changes policy inputs."""
import math

import cv2
import numpy as np
import torch


def transition_reason(episode, rows, actions, step):
    # The legacy stream rewrites its final two frames/actions to STOP. These
    # frames have no supervised next policy observation.
    if len(rows) != len(actions):
        return 'unaligned_lengths'
    if step >= len(rows) - 2 or int(actions[step]) == 0:
        return 'terminal'
    positions, yaws = episode.get('pos', []), episode.get('yaw', [])
    if len(positions) != len(rows) or len(yaws) != len(rows):
        return 'missing_pose_alignment'
    p = np.asarray(positions[step:step + 2], dtype=np.float64)
    y = np.asarray(yaws[step:step + 2], dtype=np.float64)
    if p.shape != (2, 3) or y.shape != (2,) or not np.isfinite(p).all() or not np.isfinite(y).all():
        return 'invalid_pose'
    # Validate raw body poses, not camera poses (the camera has a lever arm).
    a, yaw = int(actions[step]), y[0]
    delta, turn = np.zeros(3), 0.
    if a in (1, 8, 9):
        distance = {1: 3., 8: 6., 9: 9.}[a]
        delta[:2] = distance * np.array([math.cos(yaw), math.sin(yaw)])
    elif a in (2, 3):
        turn = math.pi / 6 * (1 if a == 2 else -1)
    elif a in (4, 5):
        delta[2] = 3. if a == 4 else -3.
    elif a in (6, 7):
        delta[:2] = (3. if a == 6 else -3.) * np.array([-math.sin(yaw), math.cos(yaw)])
    else:
        return 'unknown_action'
    angular_error = (y[1] - yaw - turn + math.pi) % (2 * math.pi) - math.pi
    if np.linalg.norm(p[1] - p[0] - delta) > .25 or abs(angular_error) > math.radians(2):
        return 'pose_action_mismatch'
    return 'valid'


def load_future_depth(path, min_depth=.1, max_depth=100.):
    depth = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f'Expected an HxW future depth image: {path}')
    mask = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    # One sampling rule for both tensors keeps invalid pixels from bleeding
    # into nearby targets. Loss is computed in log metres after resizing.
    clean = np.where(mask, depth, 0.)
    small = cv2.resize(clean, (16, 16), interpolation=cv2.INTER_NEAREST)
    valid = cv2.resize(mask.astype(np.uint8), (16, 16), interpolation=cv2.INTER_NEAREST).astype(bool)
    return torch.from_numpy(small.copy())[None], torch.from_numpy(valid.copy())[None]


def future_label(dataset, episode, rows, actions, step):
    reason = transition_reason(episode, rows, actions, step)
    target, mask = torch.zeros(1, 16, 16), torch.zeros(1, 16, 16, dtype=torch.bool)
    if reason == 'valid':
        try:
            target, mask = load_future_depth(dataset._resolve_media(rows[step + 1]['depth']),
                                            dataset.depth_min, dataset.depth_max)
            if not mask.any():
                reason = 'no_valid_pixels'
        except (FileNotFoundError, KeyError):
            reason = 'missing_future_depth'
    if getattr(dataset, 'depth_target', 'future') == 'current':
        # Preserve baseline transition/terminal filtering and future availability.
        # Only replace the supervision image; current pixels use their own mask.
        # Keep the existing future_* batch interface for the unchanged model head.
        target, mask = torch.zeros_like(target), torch.zeros_like(mask)
        if reason == 'valid':
            try:
                target, mask = load_future_depth(dataset._resolve_media(rows[step]['depth']),
                                                dataset.depth_min, dataset.depth_max)
                if not mask.any():
                    reason = 'no_valid_current_pixels'
            except (FileNotFoundError, KeyError):
                reason = 'missing_current_depth'
    dataset.future_depth_counts[reason] += 1
    return dict(future_depth=target, future_depth_mask=mask,
                future_valid=torch.tensor(reason == 'valid', dtype=torch.bool))
