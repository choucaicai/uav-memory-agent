"""Two-step TBPTT training with a selectable, checkpointed model package."""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('USE_TF', '0')

import draccus
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model.model_registry import resolve_model, checkpoint_model, stamp_model_config, canonical_model_module
from model.loss_utils import distributed_loss, detached_statistics, describe_statistics, statistics_size
from data.memory_data import get_memory_dataset_and_collator



@dataclass
class FinetuneMemoryConfig:
    num_processes: int = 4
    master_port: int = 29331
    deepspeed_config: Optional[Path] = None
    dry_run: bool = False
    model_module: str = 'himem_agent'
    model_class: str = 'OpenVLAForActionPrediction'
    depth_loss_weight: float = 0.1
    depth_edge_weight: float = 0.1
    depth_target: str = 'future'  # 'current' reconstructs D_t on baseline-eligible transitions
    depth_min: float = 0.1
    depth_max: Optional[float] = None
    depth_near_distance: float = 10.0
    depth_near_weight: float = 3.0
    depth_huber_delta: float = 1.0
    vla_path: Optional[Path] = None
    raw_train_json: Optional[Path] = None
    raw_val_json: Optional[Path] = None
    raw_media_root: Optional[Path] = None
    raw_index_path: Optional[Path] = None
    raw_val_fraction: float = 0.05
    raw_camera_name: str = 'uav_on_0'
    target_pose_mode: str = 'zero'
    run_root_dir: Optional[Path] = None
    run_id_note: Optional[str] = None
    seed: int = 11
    # Number of parallel trajectory lanes globally, before gradient accumulation.
    total_batch_size: int = 16
    tbptt_steps: int = 2
    grad_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    max_grad_norm: float = 1.0
    epoches: float = 4.0
    max_steps: Optional[int] = None
    save_steps: int = 2500
    eval_step: int = 1000
    log_steps: int = 10
    use_lora: bool = True
    lora_rank: int = 512
    lora_alpha: int = 512
    lora_dropout: float = 0.0
    init_oct_model: bool = True
    gradient_checkpointing: bool = True
    fine_voxel_size: float = 2.5
    coarse_voxel_size: float = 20.0
    visit_radius: float = 20.0
    # HiRaMem Q-Former token counts (scene and frontier queries).
    num_query_tokens: int = 128
    num_frontier_tokens: int = 16
    # Memory component ablations; defaults keep the full model.
    use_fine_memory: bool = True
    use_coarse_memory: bool = True
    use_visited_memory: bool = True
    # Loads model weights only; optimizer/data stream resume is deliberately separate.
    warm_start_checkpoint: Optional[Path] = None
    attn_implementation: str = 'flash_attention_2'
    save_final: bool = True


def load_train_config(argv=None):
    """Load YAML with CLI overrides; resolve paths against the YAML directory."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Help should work before a local configuration has been created.
    if '--help' in arguments or '-h' in arguments:
        return draccus.parse(FinetuneMemoryConfig, args=arguments)
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument('--config_path', default=str(_THIS_DIR / 'configs/train.yaml'))
    selected, overrides = selector.parse_known_args(arguments)
    path = Path(os.path.expandvars(selected.config_path)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Training configuration not found: {path}. '
                                'Copy configs/train.example.yaml to configs/train.yaml '
                                'and set your paths, or pass --config_path /path/to/train.yaml.')
    cfg = draccus.parse(FinetuneMemoryConfig, config_path=path, args=overrides)
    required = ('vla_path', 'raw_train_json', 'raw_media_root', 'run_root_dir', 'deepspeed_config')
    for name in required:
        if getattr(cfg, name) is None:
            raise ValueError(f'Missing training path {name!r} in {path} or command-line overrides')
    for name in required + ('raw_val_json', 'raw_index_path', 'warm_start_checkpoint'):
        value = getattr(cfg, name)
        if value is not None:
            candidate = Path(os.path.expandvars(str(value))).expanduser()
            setattr(cfg, name, (candidate if candidate.is_absolute() else path.parent / candidate).resolve())
    return cfg


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(v, device) for v in value)
    if is_dataclass(value):
        return replace(value, **{field.name: move_to_device(getattr(value, field.name), device)
                                 for field in fields(value)})
    return value



def base_model(model):
    return model.get_base_model() if isinstance(model, PeftModel) else model


def load_memory_vla(cfg, device):
    selected = resolve_model(cfg.model_module, cfg.model_class)
    image_processor = selected.image_processor.from_pretrained(cfg.vla_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.vla_path)
    processor = selected.processor(image_processor, tokenizer)
    config_source = cfg.warm_start_checkpoint or cfg.vla_path
    depth_options = {key: value for key, value in asdict(cfg).items() if key.startswith('depth_')}
    depth_options['depth_max'] = (cfg.depth_max if cfg.depth_max is not None
                                  else getattr(selected.model, 'default_depth_max', 100.0))
    # Pass through __init__ so model-specific configuration validation runs.
    values, _ = selected.config.get_config_dict(config_source)
    if selected.future_depth:
        values.update(depth_options)
    config = selected.config.from_dict(values)
    # These counts are training-time overrides rather than pretrained VLA fields.
    config.num_query_tokens = cfg.num_query_tokens
    config.num_frontier_tokens = cfg.num_frontier_tokens
    stamp_model_config(config, cfg.model_module, cfg.model_class, selected.auxiliaries)
    warm_auxiliaries = ()
    if cfg.warm_start_checkpoint:
        previous_module, previous_class = checkpoint_model(cfg.warm_start_checkpoint)
        warm_auxiliaries = resolve_model(previous_module, previous_class).auxiliaries
        dropped = set(warm_auxiliaries).difference(selected.auxiliaries)
        if not dropped.issubset(getattr(selected.model, 'discarded_warm_start_auxiliaries', ())):
            raise ValueError('Warm checkpoint has auxiliary branches absent from the selected model')
        if dropped:
            print(f'Warm start: omitting removed auxiliary branches {sorted(dropped)}', flush=True)
        warm_auxiliaries = tuple(name for name in warm_auxiliaries if name in selected.auxiliaries)
    config.memory_fine_voxel_size = cfg.fine_voxel_size
    config.memory_coarse_voxel_size = cfg.coarse_voxel_size
    config.memory_visit_radius = cfg.visit_radius
    config.memory_use_fine = cfg.use_fine_memory
    config.memory_use_coarse = cfg.use_coarse_memory
    config.memory_use_visited = cfg.use_visited_memory
    config.use_cache = False
    model = selected.model.from_pretrained(
        cfg.vla_path, config=config, torch_dtype=torch.bfloat16,
        attn_implementation=cfg.attn_implementation,
    ).to(device)
    if cfg.init_oct_model and cfg.warm_start_checkpoint is None:
        model.initialize_oct_model(device=device)
        model.initialize_explore_model(device=device)
        model.to(dtype=torch.bfloat16)
    if cfg.use_lora:
        excluded = ('vision_backbone.', 'oct_model.') + tuple(name + '.' for name in selected.auxiliaries)
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and not name.startswith(excluded)
                   and not name.endswith('lm_head')]
        if cfg.warm_start_checkpoint:
            model = PeftModel.from_pretrained(model, cfg.warm_start_checkpoint, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                target_modules=targets, init_lora_weights='gaussian',
            ))
    elif cfg.warm_start_checkpoint:
        raise ValueError('warm_start_checkpoint currently expects a LoRA memory checkpoint')
    core = base_model(model)
    core.vision_backbone.requires_grad_(False)
    core.oct_model.requires_grad_(False)  # Legacy additive PE is outside the memory path.
    core.explore_model.requires_grad_(True)
    core.trajectory_memory.requires_grad_(True)
    for name in selected.auxiliaries:
        getattr(core, name).requires_grad_(True)
    # Cell hidden updates are non-parametric; keep the legacy updater only for checkpoint compatibility.
    core.trajectory_memory.updater.requires_grad_(False)
    if core.explore_model.frontier_tokens.numel() == 0:
        # frontier=0 ablation: an empty parameter never receives a gradient under DDP.
        core.explore_model.frontier_tokens.requires_grad_(False)
    if cfg.warm_start_checkpoint:
        for name in warm_auxiliaries:
            weights = torch.load(Path(cfg.warm_start_checkpoint) / f'{name}.bin', map_location='cpu', weights_only=True)
            getattr(core, name).load_state_dict(weights, strict=True)
    if cfg.gradient_checkpointing:
        core.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        core.explore_model.qformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model, processor

def make_loaders(cfg, processor, local_batch_size):
    selected = resolve_model(cfg.model_module, cfg.model_class)
    max_depth = cfg.depth_max if cfg.depth_max is not None else getattr(selected.model, 'default_depth_max', 100.0)
    options = dict(train_json=cfg.raw_train_json, val_json=cfg.raw_val_json,
                   media_root=cfg.raw_media_root, index_path=cfg.raw_index_path,
                   image_transform=processor.image_processor.apply_transform,
                   tokenizer=processor.tokenizer, local_batch_size=local_batch_size,
                   tbptt_steps=cfg.tbptt_steps, seed=cfg.seed,
                   val_fraction=cfg.raw_val_fraction, camera_name=cfg.raw_camera_name,
                   target_pose_mode=cfg.target_pose_mode,
                   future_depth=selected.future_depth,
                   depth_min=cfg.depth_min, depth_max=max_depth, depth_target=cfg.depth_target)
    train_ds, _, collator = get_memory_dataset_and_collator(train=True, **options)
    val_ds, _, _ = get_memory_dataset_and_collator(train=False, **options)
    # Dataset owns both rank sharding and lane batching. Do not shard it again in Accelerate.
    loaders = [DataLoader(ds, batch_size=None, collate_fn=collator, num_workers=0)
               for ds in (train_ds, val_ds)]
    return train_ds, val_ds, *loaders


def evaluate_memory_loss(model, loader, accelerator):
    unwrapped = accelerator.unwrap_model(model)
    core = base_model(unwrapped)
    saved_states = core._memory_states
    core._memory_states = {}
    was_training = model.training
    model.eval()
    depth = getattr(core, 'requires_future_depth', False)
    totals = torch.zeros(statistics_size(core) if depth else 2, dtype=torch.float64, device=accelerator.device)
    try:
        # ZeRO-1 replicates weights; use the unwrapped model on unequal validation streams.
        with torch.no_grad(), accelerator.autocast():
            for batch in loader:
                output = unwrapped(**move_to_device(batch, accelerator.device))
                if depth:
                    totals += detached_statistics(output)
                else:
                    totals[0] += output['loss'].double() * output['token_count']
                    totals[1] += output['token_count']
    finally:
        core._memory_states = saved_states
        model.train(was_training)
    totals = accelerator.reduce(totals, reduction='sum')
    if depth:
        stats = describe_statistics(totals)
        accelerator.print('validation components: ' + json.dumps(stats))
        loss = stats['action_loss'] + core.config.depth_loss_weight * (
            stats['depth_reg_loss'] + core.config.depth_edge_weight * stats['depth_edge_loss'])
        return loss
    return (totals[0] / totals[1]).item() if totals[1] > 0 else float('nan')


def save_memory_checkpoint(model, processor, cfg, directory, step):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    core = base_model(model)
    processor.save_pretrained(directory)
    selected = resolve_model(cfg.model_module, cfg.model_class)
    stamp_model_config(core.config, cfg.model_module, cfg.model_class, selected.auxiliaries)
    core.config.save_pretrained(directory)
    for name in selected.auxiliaries:
        torch.save({k: v.detach().cpu() for k, v in getattr(core, name).state_dict().items()}, directory / f'{name}.bin')
    (directory / 'training_config.json').write_text(json.dumps(asdict(cfg), default=str, indent=2))
    (directory / 'training_progress.json').write_text(json.dumps({'optimizer_step': step, 'weights_only': True}))
    shutil.copytree(_THIS_DIR / cfg.model_module, directory / cfg.model_module,
                    ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)


def train_memory_window(model, core, batch, optimizer, accelerator, max_grad_norm):
    local_tokens = sum(int(window['labels'][:, 1:].ne(-100).sum()) for window in batch['memory_window'])
    total_tokens = accelerator.reduce(torch.tensor(local_tokens, device=accelerator.device), reduction='sum')
    with accelerator.accumulate(model):
        with accelerator.autocast():
            output = model(**move_to_device(batch, accelerator.device))
        loss = distributed_loss(output, accelerator.num_processes,
                                lambda counts: counts.copy_(accelerator.reduce(counts, reduction='sum')))
        metrics = torch.cat((
            torch.stack((loss.detach().double() * total_tokens / accelerator.num_processes,
                         total_tokens.new_tensor(local_tokens).double())),
            detached_statistics(output),
        ))
        accelerator.backward(loss)
        core.detach_memory()
        if accelerator.sync_gradients and accelerator.state.deepspeed_plugin is None and max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    return metrics


def finetune(cfg: FinetuneMemoryConfig):
    cfg.model_module = canonical_model_module(cfg.model_module)
    if cfg.depth_target not in ('future', 'current'):
        raise ValueError('depth_target must be future or current')
    if cfg.depth_target == 'current' and cfg.model_module != 'himem_agent':
        raise ValueError('Current-depth ablation is supported by himem_agent only')
    if cfg.tbptt_steps not in (1, 2):
        raise ValueError('tbptt_steps must be 1 or 2')
    if cfg.grad_accumulation_steps < 1 or cfg.log_steps < 1:
        raise ValueError('Accumulation and logging intervals must be positive')
    if cfg.max_steps is not None and cfg.max_steps < 1:
        raise ValueError('max_steps must be positive')
    if cfg.max_steps is None and cfg.epoches <= 0:
        raise ValueError('epoches must be positive')
    if cfg.num_query_tokens < 1 or cfg.num_frontier_tokens < 0:
        raise ValueError('Scene query count must be positive and frontier count non-negative')
    if not (cfg.use_fine_memory or cfg.use_coarse_memory):
        raise ValueError('At least one of fine/coarse memory must be enabled')
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accumulation_steps, mixed_precision='bf16',
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    if cfg.total_batch_size < accelerator.num_processes or cfg.total_batch_size % accelerator.num_processes:
        raise ValueError('total_batch_size must be a positive multiple of world size')
    local_batch = cfg.total_batch_size // accelerator.num_processes
    plugin = accelerator.state.deepspeed_plugin
    if plugin is not None:
        if plugin.zero_stage > 2:
            raise ValueError('This persistent-memory training/validation path supports ZeRO stages 0/1/2')
        plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = local_batch
        # Keep gradient communication in FP32; model compute remains BF16.
        plugin.deepspeed_config['communication_data_type'] = 'fp32'
        if cfg.max_grad_norm > 0:
            plugin.deepspeed_config['gradient_clipping'] = cfg.max_grad_norm
    set_seed(cfg.seed)
    run_dir = cfg.run_root_dir / cfg.run_id_note
    exists = accelerator.reduce(torch.tensor(int((run_dir / 'training_config.json').exists()),
                                            device=accelerator.device), reduction='sum')
    if exists.item():
        raise FileExistsError(f'Choose a new run_id_note; run already exists: {run_dir}')
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'training_config.json').write_text(json.dumps(asdict(cfg), default=str, indent=2))
    accelerator.wait_for_everyone()
    model, processor = load_memory_vla(cfg, accelerator.device)
    # Serialize creation of the legacy index, which publishes through a shared .tmp path.
    with accelerator.main_process_first():
        train_ds, val_ds, train_loader, val_loader = make_loaders(cfg, processor, local_batch)
    core = base_model(model)
    core.config.norm_stats.update(train_ds.dataset_statistics)
    core.norm_stats = core.config.norm_stats
    if accelerator.is_main_process:
        (run_dir / 'dataset_statistics.json').write_text(json.dumps(train_ds.dataset_statistics, indent=2))
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    model, optimizer = accelerator.prepare(model, optimizer)
    core = base_model(accelerator.unwrap_model(model))
    accelerator.print(f'TBPTT={cfg.tbptt_steps}; global trajectory batch={cfg.total_batch_size}; '
                      f'up to {cfg.tbptt_steps * cfg.total_batch_size * cfg.grad_accumulation_steps} steps/update')
    print(f'rank={accelerator.process_index} train={train_ds.episodes.local_partition()} '
          f'val={val_ds.episodes.local_partition()}', flush=True)
    sample_budget = math.ceil(cfg.epoches * len(train_ds))
    step, samples_seen = 0, 0
    recent_sum, recent_tokens = 0.0, 0
    recent_aux = torch.zeros(statistics_size(core), dtype=torch.float64, device=accelerator.device)
    model.train()
    optimizer.zero_grad()
    # Always display optimizer steps. With an epoch budget, estimate the total
    # from observed samples/update because terminal TBPTT windows can be shorter.
    samples_per_update = cfg.tbptt_steps * cfg.total_batch_size * cfg.grad_accumulation_steps
    progress_total = cfg.max_steps if cfg.max_steps is not None else math.ceil(sample_budget / samples_per_update)
    progress_desc = 'Training' if cfg.max_steps is not None else 'Training (est. total)'
    training_started = last_update_time = time.perf_counter()
    with tqdm(
        total=progress_total, desc=progress_desc, unit='step',
        dynamic_ncols=True, mininterval=1.0, disable=not accelerator.is_main_process,
    ) as progress:
        update_metrics = torch.zeros(2 + statistics_size(core), dtype=torch.float64, device=accelerator.device)
        local_samples = 0
        for batch in train_loader:
            local_samples += sum(len(window['episode_ids']) for window in batch['memory_window'])
            update_metrics += train_memory_window(model, core, batch, optimizer, accelerator, cfg.max_grad_norm)
            if not accelerator.sync_gradients:
                continue
            metrics = accelerator.reduce(update_metrics, reduction='sum')
            update_metrics.zero_()
            samples_seen += int(accelerator.reduce(
                torch.tensor(local_samples, device=accelerator.device), reduction='sum'))
            local_samples = 0
            recent_sum += metrics[0].item()
            recent_tokens += int(metrics[1])
            if len(metrics) > 2:
                recent_aux += metrics[2:]
            step += 1
            done = step >= cfg.max_steps if cfg.max_steps is not None else samples_seen >= sample_budget
            now = time.perf_counter()
            update_seconds = now - last_update_time
            elapsed_seconds = now - training_started
            if cfg.max_steps is None:
                remaining_samples = max(0, sample_budget - samples_seen)
                progress_total = step + math.ceil(remaining_samples / (samples_seen / step))
                progress.total = progress_total
            progress.set_postfix(
                loss=f'{recent_sum / recent_tokens:.4f}',
                **{'s/step': f'{update_seconds:.2f}'}, refresh=False,
            )
            progress.update(1)
            if step % cfg.log_steps == 0 or done:
                remaining = max(0, progress_total - step)
                row = {
                    'step': step,
                    'samples_seen': samples_seen, 'loss': recent_sum / recent_tokens,
                    'elapsed_seconds': elapsed_seconds, 'update_seconds': update_seconds,
                    'samples_per_second': samples_seen / max(elapsed_seconds, 1e-9),
                    'steps_per_second': step / max(elapsed_seconds, 1e-9),
                    'estimated_total_steps': progress_total,
                    'eta_seconds': remaining * elapsed_seconds / step,
                }
                if getattr(core, 'requires_future_depth', False):
                    row.update(describe_statistics(recent_aux))
                    row['future_depth_filter_counts_local'] = dict(train_ds.future_depth_counts)
                if accelerator.is_main_process:
                    tqdm.write(json.dumps(row))
                    with (run_dir / 'metrics.jsonl').open('a') as f:
                        f.write(json.dumps(row) + '\n')
                recent_sum, recent_tokens = 0.0, 0
                recent_aux.zero_()
            if cfg.eval_step > 0 and step % cfg.eval_step == 0:
                progress.set_description('Validation', refresh=True)
                val_loss = evaluate_memory_loss(model, val_loader, accelerator)
                if accelerator.is_main_process:
                    tqdm.write(f'validation step={step} loss={val_loss}')
                progress.set_description(progress_desc, refresh=True)
            save = (cfg.save_steps > 0 and step % cfg.save_steps == 0) or (done and cfg.save_final)
            if save:
                progress.set_description('Saving checkpoint', refresh=True)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_memory_checkpoint(accelerator.unwrap_model(model), processor, cfg,
                                           run_dir / f'checkpoints-{step}', step)
                accelerator.wait_for_everyone()
                progress.set_description(progress_desc, refresh=True)
            last_update_time = time.perf_counter()
            if done:
                break
    core.reset_memory()
    accelerator.end_training()


def main(cfg: Optional[FinetuneMemoryConfig] = None):
    if cfg is None:
        cfg = load_train_config()
    if cfg.num_processes < 1 or not 0 < cfg.master_port <= 65535:
        raise ValueError('num_processes must be positive and master_port must be in 1..65535')
    if cfg.run_id_note is None:
        cfg.run_id_note = f'himem_agent-tbptt{cfg.tbptt_steps}-{datetime.now():%Y%m%d-%H%M%S}'
    if not cfg.deepspeed_config.is_file():
        raise FileNotFoundError(cfg.deepspeed_config)
    command = [sys.executable, '-u', '-m', 'accelerate.commands.launch',
               '--config_file', str(cfg.deepspeed_config), '--num_processes', str(cfg.num_processes),
               '--main_process_port', str(cfg.master_port), str(Path(__file__).resolve()),
               *sys.argv[1:], '--run_id_note', cfg.run_id_note]
    if cfg.dry_run:
        print(json.dumps({'command': command, 'training': asdict(cfg)}, default=str, indent=2))
    elif os.environ.get('UAV_MEMORY_TRAIN_WORKER') == '1' or 'RANK' in os.environ:
        finetune(cfg)
    else:
        environment = dict(os.environ, UAV_MEMORY_TRAIN_WORKER='1', PYTHONUNBUFFERED='1')
        subprocess.run(command, env=environment, check=True)


if __name__ == '__main__':
    main()
