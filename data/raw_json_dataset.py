"""Streaming episode dataset for the raw UAV-ON JSON format.

The source file is a top-level JSON array.  A byte-offset index is used so DDP
ranks can seek to disjoint episode ranges.  Only one episode and its bounded
history are materialized at a time.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import cv2
import numpy as np
import orjson
from PIL import Image
import torch
from torch.utils.data import IterableDataset, get_worker_info

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .raw_json_index import ACTION_MAP, aligned_action_ids, build_index  # noqa: E402
from action_statistics import RAW_DATASET_NAME


class RawEpisodeDataset(IterableDataset):
    """Yield complete episodes one by one from a large JSON array.

    ``episode_range`` is an exclusive ``(start, end)`` range over the index.
    The range is further partitioned across distributed ranks and DataLoader
    workers.  The index itself is small and is safe to load into memory.
    """

    shards_by_rank = True

    def __init__(
        self,
        source: Path | str,
        *,
        index_path: Path | str | None = None,
        camera_name: str = "uav_on_0",
        shuffle: bool = False,
        seed: int = 11,
        episode_range: Optional[Tuple[int, int]] = None,
        auto_build_index: bool = True,
    ) -> None:
        super().__init__()
        self.source = Path(source).expanduser().resolve()
        self.camera_name = camera_name
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        if index_path is None:
            index_path = self.source.with_suffix(self.source.suffix + ".oct_agent.index.json")
        self.index_path = Path(index_path).expanduser().resolve()
        if not self.index_path.exists():
            if not auto_build_index:
                raise FileNotFoundError(f"Missing raw JSON index: {self.index_path}")
            print(f"[oct_agent] building index: {self.index_path}", flush=True)
            build_index(self.source, self.index_path, camera_name=self.camera_name)
        self.index = orjson.loads(self.index_path.read_bytes())
        source_stat = self.source.stat()
        if (
            self.index.get("source") != str(self.source)
            or self.index.get("source_size") != source_stat.st_size
            or self.index.get("source_mtime_ns") != source_stat.st_mtime_ns
        ):
            if not auto_build_index:
                raise ValueError(f"Index does not match source file: {self.index_path}")
            print(f"[oct_agent] rebuilding stale index: {self.index_path}", flush=True)
            build_index(self.source, self.index_path, camera_name=self.camera_name, force=True)
            self.index = orjson.loads(self.index_path.read_bytes())
        self.episodes = self.index["episodes"]
        self._episode_range = episode_range or (0, len(self.episodes))
        self.dataset_statistics = self.index.get("dataset_statistics", {})
        self.dataset_statistics.setdefault("raw_source", str(self.source))
        self.dataset_statistics.setdefault("index_path", str(self.index_path))
        self.dataset_statistics.setdefault("camera_name", self.camera_name)
        self.dataset_statistics.setdefault("target_pose_mode", "zero_if_missing")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _consumer() -> Tuple[int, int]:
        rank, world = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        workers = worker.num_workers if worker is not None else 1
        return rank * workers + worker_id, world * workers

    def _assigned_range(self) -> Tuple[int, int]:
        consumer, consumers = self._consumer()
        begin, end = self._episode_range
        count = max(0, end - begin)
        local_begin = begin + (count * consumer) // consumers
        local_end = begin + (count * (consumer + 1)) // consumers
        return local_begin, local_end

    def iter_episodes(self) -> Iterator[Dict[str, Any]]:
        begin, end = self._assigned_range()
        indices = list(range(begin, end))
        if self.shuffle and len(indices) > 1:
            consumer, _ = self._consumer()
            random.Random(self.seed + self.epoch * 1009 + consumer).shuffle(indices)

        # The file descriptor is shared only within this iterator and is never
        # held across workers.  Each worker/rank seeks only its assigned spans.
        with self.source.open("rb") as f:
            for index_id in indices:
                metadata = self.episodes[index_id]
                if metadata.get("num_steps", 0) <= 0 or metadata.get("error"):
                    continue
                f.seek(int(metadata["start"]))
                raw = f.read(int(metadata["end"]) - int(metadata["start"]))
                episode = orjson.loads(raw)
                episode["_episode_id"] = int(metadata["episode_id"])
                episode["_source_index"] = index_id
                yield episode

    def local_partition(self) -> Dict[str, int]:
        begin, end = self._assigned_range()
        return {
            "episode_begin": begin,
            "episode_end": end,
            "num_episodes": end - begin,
            "num_steps": sum(int(x.get("num_steps", 0)) for x in self.episodes[begin:end]),
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        yield from self.iter_episodes()

    def __len__(self) -> int:
        begin, end = self._episode_range
        return int(sum(int(x.get("num_steps", 0)) for x in self.episodes[begin:end]))


class RawJSONVLAIterableDataset(IterableDataset):
    """Turn streamed episodes into OpenVLA-compatible step samples."""

    shards_by_rank = True

    def __init__(
        self,
        source: Path | str,
        *,
        index_path: Path | str | None,
        media_root: Path | str,
        batch_transform,
        train: bool,
        seed: int = 11,
        val_fraction: float = 0.05,
        camera_name: str = "uav_on_0",
        max_history_frames: int = 24,
        target_pose_mode: str = "zero",
        shuffle: Optional[bool] = None,
        auto_build_index: bool = True,
        repeat: Optional[bool] = None,
        split_by_episode: bool = True,
        normalization_statistics: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.media_root = Path(media_root).expanduser().resolve()
        self.batch_transform = batch_transform
        self.train = bool(train)
        self.repeat = self.train if repeat is None else bool(repeat)
        self.seed = int(seed)
        self.max_history_frames = max(1, int(max_history_frames))
        self.target_pose_mode = target_pose_mode
        self.camera_name = camera_name
        self.shuffle = self.train if shuffle is None else bool(shuffle)
        val_fraction = float(val_fraction)
        if split_by_episode and not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")

        # Construct the small index-backed episode source once.  The train and
        # validation views share the index but use disjoint episode ranges.
        all_source = RawEpisodeDataset(
            source,
            index_path=index_path,
            camera_name=camera_name,
            shuffle=self.shuffle,
            seed=self.seed,
            auto_build_index=auto_build_index,
        )
        total = len(all_source.episodes)
        if split_by_episode and total < 2:
            raise ValueError("An episode train/validation split requires at least two episodes.")
        split_at = max(1, min(total - 1, int(round(total * (1.0 - val_fraction)))))
        if not split_by_episode:
            episode_range = (0, total)
        elif self.train:
            episode_range = (0, split_at)
        else:
            episode_range = (split_at, total)
        self.episodes = RawEpisodeDataset(
            source,
            index_path=all_source.index_path,
            camera_name=camera_name,
            shuffle=self.shuffle,
            seed=self.seed,
            episode_range=episode_range,
            auto_build_index=False,
        )
        stats = dict(self.episodes.dataset_statistics)
        if normalization_statistics is not None:
            stats["action"] = normalization_statistics["action"]
        self._normalization_statistics = stats
        stats["num_trajectories"] = int(episode_range[1] - episode_range[0])
        stats["num_transitions"] = len(self.episodes)
        stats["split"] = "train" if self.train else "val"
        stats["val_fraction"] = val_fraction if split_by_episode else 0.0
        stats["max_history_frames"] = self.max_history_frames
        stats["target_pose_mode"] = self.target_pose_mode
        # The legacy saver and model expect an outer dataset-name mapping.
        self.dataset_statistics = {RAW_DATASET_NAME: stats}
        self._bad_episodes = 0

    def set_epoch(self, epoch: int) -> None:
        self.episodes.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.episodes)

    def _resolve_media(self, value: Any) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = self.media_root / path
        return path

    def _load_rgb(self, value: Any) -> np.ndarray:
        path = self._resolve_media(value)
        with Image.open(path) as image:
            image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.uint8)

    def _load_depth(self, value: Any) -> np.ndarray:
        path = self._resolve_media(value)
        depth = np.load(path, allow_pickle=False)
        depth = np.asarray(depth)
        if depth.ndim > 2:
            depth = np.squeeze(depth)
            if depth.ndim > 2:
                depth = depth[..., 0]
        depth = depth.astype(np.float32, copy=False)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_NEAREST)
        return np.asarray(depth, dtype=np.float32)

    @staticmethod
    def _history_indices(length: int, maximum: int) -> list[int]:
        if length <= maximum:
            return list(range(length))
        return np.linspace(0, length - 1, maximum, dtype=np.int64).tolist()

    @staticmethod
    def _normalise_action(action: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        action_stats = stats.get("action", {})
        low = np.asarray(action_stats.get("min", np.zeros(8)), dtype=np.float32)
        high = np.asarray(action_stats.get("max", np.ones(8)), dtype=np.float32)
        denom = high - low
        out = np.zeros_like(action, dtype=np.float32)
        valid = denom > 1e-8
        out[valid] = 2.0 * (action[valid] - low[valid]) / denom[valid] - 1.0
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def _target_pose(self, episode: Dict[str, Any]) -> np.ndarray:
        target = episode.get("target")
        if target is not None:
            arr = np.asarray(target, dtype=np.float32).reshape(-1)
            if arr.size >= 3:
                return arr[:3]
        if self.target_pose_mode in {"zero", "zero_if_missing"}:
            return np.zeros(3, dtype=np.float32)
        raise ValueError(
            "Episode has no target pose. Use target_pose_mode=zero explicitly or provide train-v1-target data."
        )

    def _episode_steps(self, episode: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        image_dict = episode.get("image_dict") or {}
        rows = list(image_dict.get(self.camera_name) or [])
        raw_actions = episode.get("action") or episode.get("actions") or []
        n = min(len(rows), len(raw_actions))
        if n <= 0:
            return
        # Keep exactly the action/frame alignment recorded in the index.
        raw_actions = [int(x) for x in raw_actions[:n]]
        if n >= 2:
            action_ids = raw_actions[:-2] + [0, 0]
            frame_rows = rows[: n - 2] + [rows[n - 2], rows[n - 2]]
            history_action_ids = raw_actions[:-2]
        else:
            action_ids = raw_actions
            frame_rows = rows[:n]
            history_action_ids = raw_actions

        stats = self._normalization_statistics
        # Decode each distinct frame only once for this episode.  The cache is
        # released when the generator moves to the next episode.
        rgb_cache: list[np.ndarray] = []
        depth_cache: list[np.ndarray] = []
        params_cache: list[np.ndarray] = []
        poses_cache: list[np.ndarray] = []
        for row in frame_rows:
            rgb_cache.append(self._load_rgb(row["rgb"]))
            depth_cache.append(self._load_depth(row["depth"]))
            meta = row.get("metainfo") or {}
            quat = np.asarray(meta.get("quat_wb", [1, 0, 0, 0]), dtype=np.float32).reshape(-1)
            if quat.size < 4:
                quat = np.pad(quat, (0, 4 - quat.size))
            position = np.asarray(meta.get("camera_pos", [0, 0, 0]), dtype=np.float32).reshape(-1)
            if position.size < 3:
                position = np.pad(position, (0, 3 - position.size))
            fov = float(meta.get("fov", 90.0))
            params_cache.append(np.concatenate([quat[:4], np.asarray([fov], dtype=np.float32)]))
            poses_cache.append(position[:3].astype(np.float32))

        target_pose = self._target_pose(episode)
        instruction = str(episode.get("gpt_instruction") or episode.get("instruction") or "").strip()
        history_names: list[str] = []
        history_texts: list[str] = []
        for action_id in history_action_ids:
            history_names.append(str(action_id))
            # The existing batch transform does not consume this field, but it
            # is useful for debugging and parity with the TFDS builder.
            history_texts.append(" then ".join(history_names))

        for step_id, action_id in enumerate(action_ids):
            if action_id not in ACTION_MAP:
                raise ValueError(f"unknown action id {action_id}")
            indices = self._history_indices(step_id + 1, self.max_history_frames)
            history_images = [rgb_cache[i] for i in indices]
            history_depths = [depth_cache[i] for i in indices]
            history_params = [params_cache[i] for i in indices]
            history_poses = [poses_cache[i] for i in indices]
            raw_action = ACTION_MAP[action_id]
            yield {
                "episode_id": int(episode.get("_episode_id", -1)),
                "step_id": step_id,
                "instruction": instruction,
                "history": history_texts[step_id - 1] if step_id > 0 and step_id - 1 < len(history_texts) else "",
                "image": rgb_cache[step_id],
                "camera_images": history_images,
                "camera_depths": history_depths,
                "camera_params": history_params,
                "camera_poses": history_poses,
                "target_pose": target_pose,
                "action": self._normalise_action(raw_action, stats),
                "action_id": action_id,
                "is_last": step_id == len(action_ids) - 1,
            }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # RLDS training datasets are repeated indefinitely; reproduce that
        # behavior so the unchanged finetune loop can run for multiple epochs.
        epoch = self.episodes.epoch
        while True:
            self.episodes.set_epoch(epoch)
            yielded = 0
            for episode in self.episodes.iter_episodes():
                try:
                    for step in self._episode_steps(episode):
                        sample = self.batch_transform(step)
                        yielded += 1
                        yield sample
                except Exception as exc:
                    self._bad_episodes += 1
                    print(
                        f"[oct_agent] skip episode {episode.get('_episode_id', '?')}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
            if not self.repeat:
                break
            if yielded == 0:
                raise RuntimeError("This rank has no readable training steps; cannot repeat an empty stream.")
            epoch += 1
