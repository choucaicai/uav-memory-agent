"""Merge HiRaMem depth adapters and auxiliary weights into a full checkpoint."""
import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from model_registry import checkpoint_model


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_checkpoint(run_dir, checkpoint, adapter_dir):
    requested = Path(checkpoint)
    candidates = [requested] if requested.is_absolute() else [
        run_dir / requested, run_dir / adapter_dir / requested,
    ]
    matches = [p.resolve() for p in candidates if (p / 'adapter_config.json').is_file()]
    if len(matches) != 1:
        raise ValueError(f'Expected one adapter checkpoint, found {matches}; searched {candidates}')
    return matches[0]


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy()).hexdigest()


def state_manifest(model):
    result = {}
    for name, tensor in model.state_dict().items():
        if 'lora_' in name:
            raise ValueError(f'Unmerged adapter remains: {name}')
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f'Non-finite tensor: {name}')
        result[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'sha256': tensor_hash(tensor)}
    return result


def check_loading(info, allowed_missing=()):
    missing = [k for k in info['missing_keys'] if not k.startswith(allowed_missing)]
    if missing or info['unexpected_keys'] or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError(f'Unexpected model loading differences: {info}')


def smoke_memory(model, device):
    """Exercise the vision, two recurrent memory steps, Q-Former, and language model."""
    model.eval().to(device)
    model.reset_memory()
    ids = torch.tensor([[1, 15043, 29892, 2]], device=device)
    labels = ids.clone()
    labels[:, :2] = -100
    depth_targets = {}
    if getattr(model, 'requires_future_depth', False):
        depth_targets = dict(future_depth=torch.full((1, 1, 16, 16), 10., device=device),
                             future_depth_mask=torch.ones((1, 1, 16, 16), dtype=torch.bool, device=device),
                             future_valid=torch.ones(1, dtype=torch.bool, device=device))
    pixels = torch.zeros(1, 6, 224, 224, dtype=torch.bfloat16, device=device)
    features = torch.zeros(1, 256, model.vision_backbone.embed_dim, dtype=torch.bfloat16, device=device)
    depths = torch.full((1, 224, 224), 10.0, device=device)
    params = torch.tensor([[1.0, 0.0, 0.0, 0.0, 90.0]], device=device)
    losses = []
    with torch.inference_mode():
        for step in range(2):
            output = model(
                input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                pixel_values=pixels, camera_image_features=[features],
                camera_depths=[depths], camera_params=[params],
                camera_poses=[torch.tensor([[float(step), 0., 0.]], device=device)],
                episode_ids=['merge-smoke'], step_ids=[step], use_cache=False,
                return_dict=True, **depth_targets,
            )
            if not torch.isfinite(output.logits).all() or not torch.isfinite(output.loss):
                raise ValueError(f'Non-finite memory inference at step {step}')
            losses.append(output.loss.item())
    state = model._memory_states['merge-smoke']
    result = {'steps': 2, 'losses': losses, 'last_step': state.step_id,
              'fine_cells': len(state.fine.keys), 'coarse_cells': len(state.coarse.keys)}
    # Disabled memory levels (component ablations) are expected to stay empty.
    need_fine = getattr(model.config, 'memory_use_fine', True)
    need_coarse = getattr(model.config, 'memory_use_coarse', True)
    if (state.step_id != 1 or (need_fine and not len(state.fine.keys))
            or (need_coarse and not len(state.coarse.keys))):
        raise ValueError(f'Memory did not advance: {result}')
    model.reset_memory()
    return result


def merge_hiramem(args, checkpoint, output, base_path):
    global torch
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    import peft
    import transformers
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(11)
    # Load the actual code saved by this checkpoint, not a later working copy.
    sys.path.insert(0, str(checkpoint))
    package, class_name = checkpoint_model(checkpoint)
    model_module = importlib.import_module(f'{package}.modeling_prismatic')
    processing = importlib.import_module(f'{package}.processing_prismatic')
    if not Path(model_module.__file__).resolve().is_relative_to(checkpoint):
        raise ValueError(f'Model code did not load from checkpoint: {model_module.__file__}')
    model_class = getattr(model_module, class_name)
    config = model_class.config_class.from_pretrained(checkpoint, local_files_only=True)
    auxiliaries = getattr(model_class, 'auxiliary_module_names', ('explore_model', 'trajectory_memory'))
    statistics = read_json(Path(args.run_dir) / 'dataset_statistics.json')
    config.norm_stats = dict(config.norm_stats or {})
    config.norm_stats.update(statistics)
    config.use_cache = False
    device = args.device
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Loading base={base_path}, checkpoint={checkpoint}, device={device}', flush=True)
    model, loading_info = model_class.from_pretrained(
        base_path, config=config, torch_dtype=torch.bfloat16,
        attn_implementation='eager', local_files_only=True, output_loading_info=True,
    )
    # Base OpenVLA has no HiRaMem branches. Both active branches are restored below.
    # oct_model weights are unused by HiRaMem's non-parametric unproject_current.
    check_loading(loading_info, ('oct_model.',) + tuple(name + '.' for name in auxiliaries))
    module_counts = {}
    for name in auxiliaries:
        weights = torch.load(checkpoint / f'{name}.bin', map_location='cpu', weights_only=True)
        branch = getattr(model, name)
        branch.load_state_dict(weights, strict=True)
        for key, value in branch.state_dict().items():
            if not torch.equal(value, weights[key].to(value.dtype)):
                raise ValueError(f'Auxiliary weight mismatch: {name}.{key}')
        module_counts[name] = len(weights)
        del weights, branch, value
    model = model.to(device)
    adapted = PeftModel.from_pretrained(model, checkpoint, is_trainable=False, local_files_only=True)
    actual = get_peft_model_state_dict(adapted)
    expected = load_file(str(checkpoint / 'adapter_model.safetensors'))
    if set(actual) != set(expected):
        raise ValueError(f'Adapter key mismatch: missing={set(expected)-set(actual)}, extra={set(actual)-set(expected)}')
    for key, value in actual.items():
        if not torch.equal(value.detach().cpu(), expected[key].to(value.dtype)):
            raise ValueError(f'Adapter weight mismatch: {key}')
    adapter_count = len(expected)
    del actual, expected, value
    # Check the merge equation on the projector, attention, and feed-forward paths.
    equation_checks = {}
    for name in ('projector.fc1', 'language_model.model.layers.0.self_attn.q_proj',
                 'language_model.model.layers.0.mlp.down_proj'):
        layer = adapted.get_base_model().get_submodule(name)
        weight = layer.get_base_layer().weight.detach().clone()
        weight += layer.get_delta_weight('default').to(weight.dtype)
        equation_checks[name] = weight
    print(f'Merging {adapter_count} adapter tensors; restored {module_counts}', flush=True)
    merged = adapted.merge_and_unload(safe_merge=True)
    for name, weight in equation_checks.items():
        if not torch.equal(merged.get_submodule(name).weight, weight):
            raise ValueError(f'Merge equation mismatch: {name}')
    checked_names = list(equation_checks)
    del equation_checks, layer, weight, adapted, model
    merged.eval().to('cpu')
    manifest = state_manifest(merged)
    print(f'Validated {len(manifest)} finite merged tensors; saving to {output}', flush=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}.merging-', dir=output.parent))
    # On failure keep this uniquely named temporary directory for diagnosis.
    merged.save_pretrained(temporary, safe_serialization=True, max_shard_size=args.max_shard_size)
    processor = processing.PrismaticProcessor(
        processing.PrismaticImageProcessor.from_pretrained(checkpoint, local_files_only=True),
        AutoTokenizer.from_pretrained(checkpoint, local_files_only=True),
    )
    processor.save_pretrained(temporary)
    shutil.copytree(checkpoint / package, temporary / package,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('training_config.json', 'training_progress.json'):
        shutil.copy2(checkpoint / name, temporary / name)
    shutil.copy2(Path(args.run_dir) / 'dataset_statistics.json', temporary / 'dataset_statistics.json')
    (temporary / 'weights_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    del merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('Reloading exported model and checking every tensor SHA-256', flush=True)
    reloaded, reload_info = model_class.from_pretrained(
        temporary, torch_dtype=torch.bfloat16, attn_implementation='eager',
        local_files_only=True, output_loading_info=True,
    )
    check_loading(reload_info)
    if state_manifest(reloaded) != manifest:
        raise ValueError(f'Serialized model differs from merged model: {temporary}')
    for key in ('memory_fine_voxel_size', 'memory_coarse_voxel_size', 'memory_visit_radius',
                'memory_use_fine', 'memory_use_coarse', 'memory_use_visited'):
        if getattr(reloaded.config, key) != getattr(config, key):
            raise ValueError(f'Memory configuration mismatch: {key}')
    if reloaded.config.norm_stats != config.norm_stats:
        raise ValueError('Normalization statistics changed during export')
    print('Running two-step multimodal memory inference', flush=True)
    smoke = smoke_memory(reloaded, device)
    report = {
        'status': 'complete', 'created_at': datetime.now(timezone.utc).isoformat(),
        'base_vla': str(base_path), 'checkpoint': str(checkpoint), 'output': str(output),
        'model_class': f'{package}.modeling_prismatic.{class_name}',
        'torch': torch.__version__, 'peft': peft.__version__, 'transformers': transformers.__version__,
        'adapter_tensors_verified': adapter_count, 'auxiliary_tensors_verified': module_counts,
        'merge_equation_layers_verified': checked_names,
        'serialized_tensors_verified': len(manifest), 'reload_missing_keys': sorted(reload_info['missing_keys']),
        'reload_unexpected_keys': sorted(reload_info['unexpected_keys']), 'memory_inference': smoke,
        'base_missing_keys': sorted(loading_info['missing_keys']),
        'legacy_oct_model_note': 'Unused additive-PE weights are initialized; HiRaMem only uses unproject_current geometry.',
    }
    (temporary / 'merge_report.json').write_text(json.dumps(report, indent=2) + '\n')
    (temporary / 'README.md').write_text(
        '# Merged HiRaMem model\n\n'
        'LoRA, explore_model, and trajectory_memory are included in model-*.safetensors.\n'
        'Use the saved HiRaMem implementation (the legacy OpenVLA class does not implement memory):\n\n'
        '```python\nimport sys\nimport torch\n'
        f'model_dir = {str(output)!r}\n'
        'sys.path.insert(0, model_dir)\n'
        f'from {package}.modeling_prismatic import {class_name}\n'
        f'model = {class_name}.from_pretrained(\n'
        '    model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager"\n'
        ').eval().cuda()\n```\n\n'
        'Provide episode_ids and consecutive step_ids; call reset_memory between episodes.\n'
        'merge_report.json records reload, exact weight, and two-step inference checks.\n'
    )
    if output.exists():
        raise FileExistsError(f'Refusing to replace existing output: {output}')
    temporary.rename(output)
    print(f'MERGE_COMPLETE {output}\n{json.dumps(smoke)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vla_path', default=None)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--merge_lora_path', required=True)
    parser.add_argument('--adapter_dir', default='adapters')
    parser.add_argument('--init_oct_model', default='True')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num_threads', type=int, default=8)
    parser.add_argument('--max_shard_size', default='5GB')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    checkpoint = resolve_checkpoint(run_dir, args.merge_lora_path, args.adapter_dir)
    output = run_dir / f'full-{checkpoint.name}'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if not (checkpoint / 'trajectory_memory.bin').is_file():
        raise FileNotFoundError(checkpoint / 'trajectory_memory.bin')
    package, _ = checkpoint_model(checkpoint)
    required = ['adapter_model.safetensors', 'explore_model.bin', 'config.json',
                'training_config.json', 'training_progress.json', 'preprocessor_config.json',
                f'{package}/modeling_prismatic.py', f'{package}/trajectory_memory.py']
    required += [f'{name}.bin' for name in read_json(checkpoint / 'config.json').get(
        'memory_auxiliary_modules', ('explore_model', 'trajectory_memory'))]
    for name in required:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    base_path = Path(args.vla_path or read_json(checkpoint / 'training_config.json')['vla_path']).resolve()
    if not (base_path / 'config.json').exists():
        raise FileNotFoundError(base_path)
    if not (run_dir / 'dataset_statistics.json').is_file():
        raise FileNotFoundError(run_dir / 'dataset_statistics.json')
    if args.dry_run:
        print(json.dumps({'checkpoint': str(checkpoint), 'base_vla': str(base_path), 'output': str(output)}, indent=2))
        return
    merge_hiramem(args, checkpoint, output, base_path)


if __name__ == '__main__':
    main()
