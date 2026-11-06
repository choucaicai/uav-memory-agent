"""OpenVLA batch adapter for raw UAV-ON RGBD episodes."""
from __future__ import annotations

import sys
import numpy as np
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.action_tokenizer import ActionTokenizer  # noqa: E402
from vln_datasets.data_utils import PaddedCollatorForActionPrediction  # noqa: E402
from vln_datasets.dataset import SingleDepthRLDSBatchTransform  # noqa: E402

from .raw_json_dataset import RawEpisodeDataset, RawJSONVLAIterableDataset  # noqa: E402
from action_statistics import RAW_DATASET_NAME


class RawDepthRLDSBatchTransform:
    """Adapt one raw step to the historical SingleDepthRLDSBatchTransform.

    Keeping the final transform unchanged preserves prompt construction,
    action-token formatting and image preprocessing used by the existing v13
    training runs.
    """

    def __init__(self, action_tokenizer, tokenizer, image_transform, predict_stop_token=True):
        self.delegate = SingleDepthRLDSBatchTransform(
            action_tokenizer,
            tokenizer,
            image_transform,
            predict_stop_token=predict_stop_token,
        )

    def __call__(self, step):
        # The historical transform expects an RLDS-like batch with an outer
        # batch dimension of one.  All actual episode data remains bounded by
        # one episode in RawJSONVLAIterableDataset.
        rlds_batch = {
            "dataset_name": RAW_DATASET_NAME,
            "action": np.asarray([step["action"]], dtype=np.float32),
            "task": {"language_instruction": step["instruction"].encode("utf-8")},
            "observation": {
                "image_0": np.asarray([step["image"]], dtype=np.uint8),
                "camera_0_image": np.asarray([step["camera_images"]], dtype=np.uint8),
                "camera_0_depth": np.asarray([step["camera_depths"]], dtype=np.float32),
                "camera_0_camera_pos": np.asarray([step["camera_poses"]], dtype=np.float32),
                "camera_0_quat_wb": np.asarray(
                    [[params[:4] for params in step["camera_params"]]], dtype=np.float32
                ),
                "camera_0_fov": np.asarray(
                    [[float(params[4]) for params in step["camera_params"]]], dtype=np.float32
                ),
                "target_pose": np.asarray([step["target_pose"]], dtype=np.float32),
            },
        }
        transformed = self.delegate(rlds_batch)
        transformed["episode_id"] = step["episode_id"]
        transformed["step_id"] = step["step_id"]
        transformed["action_id"] = step["action_id"]
        return transformed


def get_raw_vla_dataset_and_collator(
    *,
    train_json: Path,
    val_json: Optional[Path],
    media_root: Path,
    index_path: Optional[Path],
    image_transform,
    tokenizer,
    default_image_resolution,
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 1000,
    train: bool = True,
    seed: int = 11,
    val_fraction: float = 0.05,
    camera_name: str = "uav_on_0",
    max_history_frames: int = 24,
    target_pose_mode: str = "zero",
):
    """Build a dataset compatible with the existing finetune loop.

    ``val_json`` may point to a separate file.  If omitted, the last
    ``val_fraction`` of ``train_json`` episodes is used for validation.
    ``shuffle_buffer_size`` is retained for API parity; shuffling happens at
    episode level because the source is indexed by episode.
    """
    del default_image_resolution, shuffle_buffer_size
    normalization_statistics = None
    if not train and val_json is not None:
        source = Path(val_json)
        # A separate validation file gets its own index unless the caller
        # explicitly supplied a matching path.
        source_index = index_path if Path(train_json).resolve() == source.resolve() else None
        normalization_statistics = RawEpisodeDataset(
            train_json, index_path=index_path, camera_name=camera_name,
        ).dataset_statistics
    else:
        source = Path(train_json)
        source_index = index_path

    action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = RawDepthRLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_transform,
        predict_stop_token=predict_stop_token,
    )
    dataset = RawJSONVLAIterableDataset(
        source,
        index_path=source_index,
        media_root=media_root,
        batch_transform=batch_transform,
        train=train,
        seed=seed,
        val_fraction=val_fraction,
        split_by_episode=val_json is None,
        normalization_statistics=normalization_statistics,
        camera_name=camera_name,
        max_history_frames=max_history_frames,
        target_pose_mode=target_pose_mode,
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        padding_side=padding_side,
    )
    return dataset, action_tokenizer, collator
