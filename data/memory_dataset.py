"""Current-frame episode streams arranged into persistent lanes and 2-step windows."""
from __future__ import annotations
import numpy as np
import torch
from collections import Counter
from .future_depth import future_label
from torch.utils.data import get_worker_info
from .raw_json_dataset import RawJSONVLAIterableDataset, ACTION_MAP


class MemoryEpisodeDataset(RawJSONVLAIterableDataset):
    def __init__(self, *args, local_batch_size=1, tbptt_steps=2, future_depth=False,
                 depth_min=.1, depth_max=100., depth_target='future', **kwargs):
        if depth_target not in ('future', 'current'):
            raise ValueError('depth_target must be future or current')
        if depth_target == 'current' and not future_depth:
            raise ValueError('Current-depth labels require the depth-only auxiliary task')
        if tbptt_steps not in (1, 2) or local_batch_size < 1:
            raise ValueError("Use tbptt_steps=1 or 2 and a positive trajectory batch size")
        kwargs['max_history_frames'] = 1
        super().__init__(*args, **kwargs)
        self.local_batch_size = int(local_batch_size)
        self.tbptt_steps = tbptt_steps
        self.future_depth = future_depth
        self.depth_target = depth_target
        self.depth_min, self.depth_max = depth_min, depth_max
        self.future_depth_counts = Counter()

    def _episode_steps(self, episode):
        rows = list((episode.get('image_dict') or {}).get(self.camera_name) or [])
        actions = episode.get('action') or episode.get('actions') or []
        original_rows = rows
        n = min(len(rows), len(actions))
        if n == 0:
            return
        # Preserve the original dataset's final two STOP labels and frame alignment.
        action_ids = [int(x) for x in actions[:n]]
        if n >= 2:
            action_ids = action_ids[:-2] + [0, 0]
            rows = rows[:n - 2] + [rows[n - 2], rows[n - 2]]
        else:
            rows = rows[:n]
        instruction = str(episode.get('gpt_instruction') or episode.get('instruction') or '').strip()
        target = self._target_pose(episode)
        for step_id, (row, action_id) in enumerate(zip(rows, action_ids)):
            if action_id not in ACTION_MAP:
                raise ValueError(f'Unknown action id {action_id}')
            meta = row.get('metainfo') or {}
            quat = np.asarray(meta.get('quat_wb', [1, 0, 0, 0]), dtype=np.float32).reshape(-1)
            pose = np.asarray(meta.get('camera_pos', [0, 0, 0]), dtype=np.float32).reshape(-1)
            if quat.size < 4:
                quat = np.pad(quat, (0, 4 - quat.size))
            if pose.size < 3:
                pose = np.pad(pose, (0, 3 - pose.size))
            params = np.concatenate([quat[:4], np.array([meta.get('fov', 90.0)], dtype=np.float32)])
            image, depth = self._load_rgb(row['rgb']), self._load_depth(row['depth'])
            sample = dict(
                episode_id=episode['_episode_id'], memory_id=episode['_memory_id'],
                step_id=step_id, is_last=step_id == n - 1, instruction=instruction,
                image=image, camera_images=[image], camera_depths=[depth],
                camera_params=[params], camera_poses=[pose[:3]], target_pose=target,
                action=self._normalise_action(ACTION_MAP[action_id], self._normalization_statistics),
                action_id=action_id,
            )
            if self.future_depth:
                sample.update(future_label(self, episode, original_rows, actions, step_id))
            yield sample

    def _episode_stream(self):
        epoch = self.episodes.epoch
        consumer, _ = self.episodes._consumer()
        while True:
            self.episodes.set_epoch(epoch)
            yielded = False
            for episode in self.episodes.iter_episodes():
                yielded = True
                episode['_memory_id'] = f'{consumer}:{epoch}:{episode["_source_index"]}'
                yield episode
            if not self.repeat:
                return
            if not yielded:
                raise RuntimeError('This rank owns no valid episodes; reduce world size or change the split')
            epoch += 1

    def __iter__(self):
        if get_worker_info() is not None:
            raise ValueError('Persistent memory lanes require DataLoader num_workers=0')
        source = self._episode_stream()
        lanes = [None] * self.local_batch_size
        while True:
            for lane in range(len(lanes)):
                if lanes[lane] is None:
                    episode = next(source, None)
                    if episode is not None:
                        lanes[lane] = iter(self._episode_steps(episode))
            if all(lane is None for lane in lanes):
                return
            steps = [[] for _ in range(self.tbptt_steps)]
            for lane, iterator in enumerate(lanes):
                if iterator is None:
                    continue
                for time in range(self.tbptt_steps):
                    raw = next(iterator, None)
                    if raw is None:
                        lanes[lane] = None
                        break
                    sample = self.batch_transform(raw)
                    sample.update(memory_id=raw['memory_id'], step_id=raw['step_id'], is_last=raw['is_last'])
                    if self.future_depth:
                        sample.update({key: raw[key] for key in ('future_depth', 'future_depth_mask', 'future_valid')})
                    steps[time].append(sample)
                    if raw['is_last']:
                        lanes[lane] = None
                        break
            steps = [items for items in steps if items]
            if steps:
                yield steps
