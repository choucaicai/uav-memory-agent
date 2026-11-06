"""Collate time-major windows without flattening or shuffling trajectory steps."""
import torch
from vln_datasets.data_utils import PaddedCollatorForActionPrediction


class MemoryWindowCollator:
    def __init__(self, tokenizer):
        self.step_collator = PaddedCollatorForActionPrediction(
            tokenizer.model_max_length, tokenizer.pad_token_id, padding_side='right')

    def __call__(self, steps):
        # DataLoader uses batch_size=None: dataset already groups trajectory lanes.
        batches = []
        for samples in steps:
            batch = self.step_collator(samples)
            batch.pop('dataset_names', None)
            batch['episode_ids'] = [s['memory_id'] for s in samples]
            batch['step_ids'] = torch.tensor([s['step_id'] for s in samples], dtype=torch.long)
            batch['episode_ends'] = torch.tensor([s['is_last'] for s in samples], dtype=torch.bool)
            if 'future_depth' in samples[0]:
                for key in ('future_depth', 'future_depth_mask', 'future_valid'):
                    batch[key] = torch.stack([s[key] for s in samples])
            batches.append(batch)
        return {'memory_window': batches}
