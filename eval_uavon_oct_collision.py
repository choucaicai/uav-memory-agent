"""Collision-terminating UAV-ON evaluation for merged HiRaMem models.

torchrun supplies rank metadata only: workers own disjoint episodes and independent
models/simulators. There are no DDP collectives or DeepSpeed training wrappers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
COLLISION_PROTOCOL = 'terminate_and_fail_external_physics_v1'
sys.path.insert(0, str(ROOT))
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import airsim
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer

from action_statistics import resolve_action_stats_key
from data.raw_json_index import ACTION_MAP
from collision_motion import CollisionMotion, read_pose
from eval_utils import (
    ENV_DICT, AIRSIM_CONFIG_DIR, RUNTIME_CONFIG, AirsimTrajRecorder, calculate_distance_uavon,
    convert_to_action_id, getPoseAfterMakeAction, to_eularian_angles,
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def episode_key(row):
    return str(row['map_name']), str(row['episode_id'])


def result_path(folder, row):
    name = '--'.join(quote(part, safe='') for part in episode_key(row))
    return Path(folder) / 'temp' / f'{name}.json'


def select_episodes(cfg):
    rows = json.loads(Path(cfg.eval_dataset).read_text())
    keys = [episode_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Dataset contains duplicate (map_name, episode_id) keys')
    grouped = defaultdict(list)
    for row in rows:
        if cfg.env_names and row['map_name'] not in cfg.env_names:
            continue
        grouped[row['map_name']].append(row)
    if cfg.env_names and set(cfg.env_names) - set(grouped):
        raise ValueError(f'Unknown requested maps: {set(cfg.env_names) - set(grouped)}')
    selected = []
    for group in grouped.values():
        selected.extend(group[:cfg.eval_samples_per_env] if cfg.eval_samples_per_env else group)
    if cfg.max_episodes is not None:
        selected = selected[:cfg.max_episodes]
    if not selected:
        raise ValueError('No selected evaluation episodes')
    return selected


def partition_episodes(rows, rank, world_size):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['map_name']].append(row)
    return {name: group[rank::world_size] for name, group in grouped.items() if group[rank::world_size]}


def compatible_code_hash(name, path):
    # Infrastructure-only revisions (e.g. placement retries) are mapped to the
    # hash of the code they are equivalent to, so earlier results stay resumable.
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    aliases_path = ROOT / 'code_compatibility.json'
    aliases = json.loads(aliases_path.read_text()) if aliases_path.exists() else {}
    return aliases.get(name, {}).get(digest, digest)


def evaluation_identity(cfg):
    # Worker count/ports may change when resuming; evaluation semantics may not.
    values = {key: getattr(cfg, key) for key in (
        'eval_max_steps', 'success_distance', 'unnorm_key', 'attn_implementation',
        'camera_name', 'seed', 'settle_seconds', 'eval_samples_per_env', 'max_episodes', 'env_names',
        'collision_substep_m', 'collision_yaw_step_degrees', 'collision_poll_seconds',
        'motion_timeout_seconds', 'position_tolerance', 'yaw_tolerance_degrees',
    )}
    values.update(
        model_path=str(Path(cfg.model_path).resolve()),
        eval_dataset=str(Path(cfg.eval_dataset).resolve()),
        dataset_sha256=hashlib.sha256(Path(cfg.eval_dataset).read_bytes()).hexdigest(),
        evaluator_sha256=compatible_code_hash('evaluator_sha256', Path(__file__)),
        collision_motion_sha256=compatible_code_hash('collision_motion_sha256', ROOT / 'collision_motion.py'),
        simulator_settings_sha256=hashlib.sha256((AIRSIM_CONFIG_DIR / 'base_settings_512.json').read_bytes()).hexdigest(),
        collision_protocol=COLLISION_PROTOCOL,
        model_manifest_sha256=hashlib.sha256((Path(cfg.model_path) / 'weights_manifest.json').read_bytes()).hexdigest(),
    )
    return values


def ensure_identity(cfg):
    folder = Path(cfg.output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    identity = evaluation_identity(cfg)
    with (folder / '.eval.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = folder / 'eval_config.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous != identity:
                raise ValueError(f'Output folder belongs to a different evaluation: {folder}')
        atomic_json(path, identity)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def completed_result(cfg, row, signature):
    path = result_path(cfg.output_folder, row)
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if (saved.get('status') != 'complete' or episode_key(saved) != episode_key(row)
            or saved.get('evaluation_signature') != signature):
        raise ValueError(f'Incompatible or incomplete episode result: {path}')
    return saved


def build_prompt(instruction):
    # This is the same builder and lowercasing as SingleDepthRLDSBatchTransform.
    from model.prompt_llama2 import LLaMa2ChatPromptBuilder
    builder = LLaMa2ChatPromptBuilder('prismatic')
    builder.add_turn('human', f'What action should the robot take to {instruction.strip().lower()}?')
    return builder.get_prompt()


def load_policy(cfg, device):
    path = Path(cfg.model_path).resolve()
    sys.path.insert(0, str(path))
    saved_config = json.loads((path / 'config.json').read_text())
    package = saved_config.get('memory_model_module', 'model_hiramem_depth')
    class_name = saved_config.get('memory_model_class', 'OpenVLAForActionPrediction')
    modeling = importlib.import_module(f'{package}.modeling_prismatic')
    processing = importlib.import_module(f'{package}.processing_prismatic')
    if not Path(modeling.__file__).resolve().is_relative_to(path):
        raise ValueError(f'Expected checkpoint model code, got {modeling.__file__}')
    policy, info = getattr(modeling, class_name).from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation=cfg.attn_implementation,
        low_cpu_mem_usage=True, local_files_only=True, output_loading_info=True,
    )
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Merged model is incomplete: {info}')
    policy = policy.eval().to(device)
    policy.requires_grad_(False)
    policy.config.use_cache = True
    policy.config.text_config.use_cache = True
    policy.generation_config.use_cache = True
    processor = processing.PrismaticProcessor(
        processing.PrismaticImageProcessor.from_pretrained(path, local_files_only=True),
        AutoTokenizer.from_pretrained(path, local_files_only=True),
    )
    key = resolve_action_stats_key(policy.norm_stats, cfg.unnorm_key)
    if policy.get_action_dim(key) != 8:
        raise ValueError(f'UAV-ON requires eight action components, got {policy.get_action_dim(key)}')
    print(f'MODEL_READY rank={os.environ.get("RANK", "0")} checkpoint={path.name} '
          f'unnorm_key={key} memory={policy.config.memory_fine_voxel_size}/'
          f'{policy.config.memory_coarse_voxel_size}/{policy.config.memory_visit_radius}', flush=True)
    return policy, processor, key


@torch.inference_mode()
def get_action(policy, processor, observation, instruction, episode_id, step_id, device, unnorm_key):
    image = Image.fromarray(observation['rgb']).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
    depth = np.asarray(observation['depth'], dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_NEAREST)
    meta = observation['metainfo']
    params = np.asarray(list(meta['quat_wb']) + [float(meta['fov'])], dtype=np.float32)
    pose = np.asarray(meta['camera_pos'], dtype=np.float32)
    if params.shape != (5,) or pose.shape != (3,) or not np.isfinite(params).all() or not np.isfinite(pose).all():
        raise ValueError(f'Invalid camera geometry: params={params}, pose={pose}')
    inputs = processor(build_prompt(instruction), image).to(device, dtype=torch.bfloat16)
    # predict_action appends this token itself but does not extend attention_mask.
    # Append both here so generation has a correctly aligned prompt and mask.
    if inputs['input_ids'][0, -1].item() != 29871:
        inputs['input_ids'] = torch.cat((inputs['input_ids'], inputs['input_ids'].new_tensor([[29871]])), dim=1)
        inputs['attention_mask'] = torch.cat((inputs['attention_mask'], inputs['attention_mask'].new_ones((1, 1))), dim=1)
    inputs.update(
        camera_images=inputs['pixel_values'].unsqueeze(1),
        camera_depths=torch.from_numpy(depth).to(device)[None, None],
        camera_params=torch.from_numpy(params).to(device)[None, None],
        camera_poses=torch.from_numpy(pose).to(device)[None, None],
        episode_ids=[episode_id], step_ids=[step_id],
    )
    raw = policy.predict_action(
        **inputs, unnorm_key=unnorm_key, do_sample=False, use_cache=True,
        min_new_tokens=policy.get_action_dim(unnorm_key),
    )
    if np.asarray(raw).shape != (8,) or not np.isfinite(raw).all():
        raise ValueError(f'Invalid action output: {raw}')
    rounded = np.rint(raw).astype(int)
    action_id = int(convert_to_action_id(rounded))
    valid_action = bool(np.array_equal(rounded, ACTION_MAP[action_id]))
    memory = policy._memory_states[episode_id]
    if len(policy._memory_states) != 1 or memory.step_id != step_id:
        raise ValueError('Episode memory leaked or did not advance exactly once')
    return action_id, {
        'raw': raw.tolist(), 'rounded': rounded.tolist(), 'valid_action': valid_action,
        'fine_cells': len(memory.fine.keys), 'coarse_cells': len(memory.coarse.keys),
    }


class ManagedSimulator(AirsimTrajRecorder):
    """Reuse established capture/pose code; own and bound this process's lifecycle."""
    def __init__(self, env_name, port, gpu_id, folder, startup_timeout):
        self.process = None
        self.log_file = None
        self._client = None
        self.env_name = env_name
        self.airsim_port = port
        with socket.socket() as probe:
            # A just-closed AirSim server may leave TIME_WAIT connections. These
            # are not live listeners; permit reuse while still rejecting a bound listener.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', port))
        directory = Path(folder) / 'simulator'
        directory.mkdir(parents=True, exist_ok=True)
        settings = json.loads((AIRSIM_CONFIG_DIR / 'base_settings_512.json').read_text())
        settings['ApiServerPort'] = port
        # Official AirSim mode for discrete pose control with collision callbacks.
        settings['PhysicsEngineName'] = 'ExternalPhysicsEngine'
        for vehicle in settings['Vehicles'].values():
            vehicle.pop('EnableCollisionPassthrogh', None)  # Legacy settings typo.
            vehicle['EnableCollisions'] = True
            vehicle['EnableCollisionPassthrough'] = False
        settings_path = directory / f'settings-{port}.json'
        atomic_json(settings_path, settings)
        log_path = directory / f'{env_name}-{port}.log'
        self.log_file = log_path.open('a')
        command = ['bash', ENV_DICT[env_name], '-RenderOffscreen', '-NoSound', '-NoVSync',
                   f'-GraphicsAdapter={gpu_id}', f'--settings={settings_path.resolve()}']
        print(f'SIM_START map={env_name} port={port} gpu={gpu_id} log={log_path}', flush=True)
        try:
            self.process = subprocess.Popen(command, stdout=self.log_file, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True)
            deadline = time.monotonic() + startup_timeout
            last_error = None
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'Simulator exited with {self.process.returncode}; see {log_path}')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=1):
                        pass
                    self._client = airsim.MultirotorClient(port=port, timeout_value=10)
                    if not self._client.ping():
                        raise RuntimeError('AirSim ping failed')
                    self._client.enableApiControl(True)
                    self._client.armDisarm(True)
                    print(f'SIM_READY map={env_name} port={port}', flush=True)
                    return
                except Exception as exc:
                    last_error = exc
                    if self._client is not None:
                        self._client.client.close()
                        self._client = None
                    time.sleep(1)
            raise TimeoutError(f'Simulator not ready within {startup_timeout}s: {last_error}; see {log_path}')
        except BaseException:
            self.cleanup()
            raise

    def cleanup(self):
        if self._client is not None:
            try:
                self._client.client.close()
            except Exception:
                pass
            self._client = None
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
            self.process = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def __del__(self):
        self.cleanup()


def evaluate_episode(cfg, row, simulator, policy, processor, key, device, signature, rank):
    start = row['start_pose']
    _, _, yaw = to_eularian_angles(*start['start_quaternionr'])
    pose = [*start['start_position'], yaw]
    actions, details = [], []
    memory_id = f'{row["map_name"]}:{row["episode_id"]}'
    policy.reset_memory()
    started = time.monotonic()
    motion = CollisionMotion(
        simulator._client, substep_m=cfg.collision_substep_m,
        yaw_step_degrees=cfg.collision_yaw_step_degrees, poll_seconds=cfg.collision_poll_seconds,
        timeout_seconds=cfg.motion_timeout_seconds, settle_seconds=cfg.settle_seconds,
        position_tolerance=cfg.position_tolerance, yaw_tolerance_degrees=cfg.yaw_tolerance_degrees,
    )
    collision = None
    collision_step = None
    collision_phase = None
    termination = 'max_steps'
    traveled_distance = 0.0
    try:
        pose = motion.begin_episode(pose)
        trajectory = [list(pose)]
        oracle_success = int(calculate_distance_uavon(pose[:3], row['pose']) <= cfg.success_distance)
        for step in range(cfg.eval_max_steps):
            time.sleep(cfg.settle_seconds)
            collision = motion.poll_collision()
            if collision:
                pose = read_pose(simulator._client)
                trajectory[-1] = list(pose)
                collision_step, collision_phase, termination = step, 'before_observation', 'collision'
                break
            observation = simulator._capture_images(camera_names=[cfg.camera_name], capture_depth=True)[cfg.camera_name]
            action, detail = get_action(policy, processor, observation, row['description'], memory_id,
                                        step, device, key)
            actions.append(action)
            requested_pose = getPoseAfterMakeAction(pose, action)
            movement = motion.move_to(requested_pose)
            if movement['motion_timeout']:
                # A missing callback/pose update is an infrastructure error, not
                # a model failure. Do not write a completed episode for it.
                raise TimeoutError(f'Movement timed out episode={memory_id} step={step} '
                                   f'target={requested_pose} actual={movement["pose"]}')
            previous = pose
            for actual in movement['trace']:
                traveled_distance += math.dist(previous[:3], actual[:3])
                oracle_success = max(oracle_success, int(
                    calculate_distance_uavon(actual[:3], row['pose']) <= cfg.success_distance))
                previous = actual
            pose = movement['pose']
            detail.update(requested_pose=list(requested_pose), actual_pose=list(pose),
                          movement_trace=movement['trace'], collision=movement['collision'])
            details.append(detail)
            trajectory.append(list(pose))
            distance = calculate_distance_uavon(pose[:3], row['pose'])
            oracle_success = max(oracle_success, int(distance <= cfg.success_distance))
            collision = movement['collision']
            if step % cfg.log_step_interval == 0 or action == 0 or collision or step + 1 == cfg.eval_max_steps:
                print(f'STEP rank={rank} episode={memory_id} step={step} action={action} '
                      f'valid={detail["valid_action"]} distance={distance:.3f} '
                      f'cells={detail["fine_cells"]}/{detail["coarse_cells"]} collision={bool(collision)}', flush=True)
            if collision:
                collision_step, collision_phase, termination = step, 'action', 'collision'
                break
            if action == 0:
                termination = 'stop'
                break
        distance = calculate_distance_uavon(pose[:3], row['pose'])
        result = dict(row)
        result.update(
            status='complete', evaluation_signature=signature, model_path=str(Path(cfg.model_path).resolve()),
            acc=int(collision is None and distance <= cfg.success_distance), osr=oracle_success, ne=distance,
            collision_free_osr=int(collision is None and oracle_success),
            collision=bool(collision), collision_count=int(bool(collision)),
            first_collision_step=collision_step, collision_phase=collision_phase, collision_info=collision,
            termination_reason=termination, collision_protocol=COLLISION_PROTOCOL,
            traveled_distance_m=traveled_distance,
            gen_pose_list=trajectory, predicted_actions=actions, action_details=details,
            steps=len(actions), stopped=termination == 'stop',
            invalid_action_count=sum(not item['valid_action'] for item in details),
            elapsed_seconds=time.monotonic() - started, rank=rank,
        )
        atomic_json(result_path(cfg.output_folder, row), result)
        print(f'EPISODE_COMPLETE rank={rank} episode={memory_id} steps={len(actions)} '
              f'SR={result["acc"]} OSR={result["osr"]} NE={distance:.3f} '
              f'termination={termination} collision_object={collision["object_name"] if collision else None}', flush=True)
        return result
    finally:
        policy.reset_memory()


def preflight(cfg, rows):
    path = Path(cfg.model_path)
    config = json.loads((path / 'config.json').read_text())
    package = config.get('memory_model_module', 'model_hiramem_depth')
    for name in ('model.safetensors.index.json', 'weights_manifest.json', 'config.json',
                 f'{package}/modeling_prismatic.py', f'{package}/trajectory_memory.py'):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    config = json.loads((path / 'config.json').read_text())
    for name in ('memory_fine_voxel_size', 'memory_coarse_voxel_size', 'memory_visit_radius'):
        if name not in config:
            raise ValueError(f'Checkpoint is not configured for HiRaMem: missing {name}')
    resolve_action_stats_key(config['norm_stats'], cfg.unnorm_key)
    for name in {row['map_name'] for row in rows}:
        if name not in ENV_DICT or not Path(ENV_DICT[name]).is_file():
            raise FileNotFoundError(f'No simulator executable configured for {name}')
    if cfg.eval_max_steps < 1 or cfg.success_distance <= 0 or cfg.log_step_interval < 1 or cfg.settle_seconds < 0:
        raise ValueError('Invalid evaluation limits')
    if any(not math.isfinite(value) or value <= 0 for value in (
            cfg.collision_substep_m, cfg.collision_yaw_step_degrees, cfg.collision_poll_seconds,
            cfg.motion_timeout_seconds, cfg.position_tolerance, cfg.yaw_tolerance_degrees)):
        raise ValueError('Collision movement parameters must be finite and positive')
    for value in (cfg.eval_samples_per_env, cfg.max_episodes):
        if value is not None and value < 1:
            raise ValueError('Episode limits must be positive')


def summarize(cfg, rows):
    from eval_results_v2 import process_results, print_results
    signature = ensure_identity(cfg)
    groups = defaultdict(list)
    for row in rows:
        result = completed_result(cfg, row, signature)
        if result is None:
            raise ValueError(f'Missing episode result: {episode_key(row)}')
        groups[row['map_name']].append(result)
    summary = process_results(groups)
    summary.update(status='complete', expected_episodes=len(rows), evaluation_signature=signature,
                   model_path=str(Path(cfg.model_path).resolve()))
    summary['env_results'] = {
        name: {'env': name, 'result': process_results({name: group})['total_results']}
        for name, group in groups.items()
    }
    summary['invalid_action_count'] = sum(r['invalid_action_count'] for group in groups.values() for r in group)
    def collision_metrics(group):
        return {'collision_count': sum(r['collision'] for r in group),
                'collision_rate': sum(r['collision'] for r in group) / len(group),
                'collision_free_osr': sum(r['collision_free_osr'] for r in group) / len(group),
                'termination_counts': {reason: sum(r['termination_reason'] == reason for r in group)
                                       for reason in ('collision', 'stop', 'max_steps')}}
    summary['collision_protocol'] = COLLISION_PROTOCOL
    summary['total_results'].update(collision_metrics([r for group in groups.values() for r in group]))
    for name, group in groups.items():
        summary['env_results'][name]['result'].update(collision_metrics(group))
    atomic_json(Path(cfg.output_folder) / 'results.json', summary)
    print_results(summary)
    print(f'COLLISION_METRICS {json.dumps(collision_metrics([r for group in groups.values() for r in group]))}', flush=True)
    print(f'EVAL_COMPLETE checkpoint={Path(cfg.model_path).name} episodes={len(rows)} '
          f'results={Path(cfg.output_folder) / "results.json"}', flush=True)


def simulator_gpu_id(cfg, local_rank):
    if cfg.simulator_gpu_ids:
        ids = [int(value.strip()) for value in cfg.simulator_gpu_ids.split(',')]
        if local_rank >= len(ids) or min(ids) < 0:
            raise ValueError('--simulator_gpu_ids must provide a Vulkan GPU index for every local worker')
        return ids[local_rank]
    # Vulkan enumeration differs from nvidia-smi/CUDA on this server. Match
    # physical device UUIDs instead of silently rendering on another user's GPU.
    try:
        listing = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True, timeout=15)
        uuids = {index.strip(): uuid.strip().removeprefix('GPU-').lower()
                 for index, uuid in (line.split(',') for line in listing.strip().splitlines())}
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(uuids)).split(',')
        device = visible[local_rank].strip()
        uuid = uuids.get(device, device.removeprefix('GPU-').lower())
        vulkan = subprocess.check_output(['vulkaninfo', '--summary'], text=True,
                                         stderr=subprocess.DEVNULL, timeout=30)
        matches = [(int(index), value.lower()) for index, value in
                   re.findall(r'GPU(\d+):.*?deviceUUID\s*=\s*([\da-fA-F-]+)', vulkan, re.S)]
        selected = [index for index, value in matches if value.startswith(uuid)]
        if len(selected) != 1:
            raise ValueError(f'Cannot match CUDA device {device} UUID {uuid} to Vulkan')
        print(f'GPU_MAPPING local_rank={local_rank} cuda_visible={device} vulkan={selected[0]}', flush=True)
        return selected[0]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise RuntimeError('Unable to map simulator GPU; specify --simulator_gpu_ids with Vulkan indices') from exc


def run(cfg, rows):
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.set_num_threads(cfg.num_threads)
    torch.cuda.set_device(local_rank)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cuda', local_rank)
    signature = ensure_identity(cfg)
    groups = partition_episodes(rows, rank, world)
    groups = {name: [row for row in group if completed_result(cfg, row, signature) is None]
              for name, group in groups.items()}
    groups = {name: group for name, group in groups.items() if group}
    print(f'WORKER_START rank={rank}/{world} device={device} episodes={sum(map(len, groups.values()))}', flush=True)
    if not groups:
        return
    gpu_id = simulator_gpu_id(cfg, local_rank)
    policy, processor, key = load_policy(cfg, device)
    completed = 0
    for scene_index, (name, group) in enumerate(groups.items()):
        # AirSim can leave a listening/TIME_WAIT socket briefly after its wrapper
        # exits. Never reuse a worker's preceding scene port during this run.
        scene_port = cfg.airsim_default_port + scene_index * world + rank
        if scene_port > 65535:
            raise ValueError(f'Simulator port is out of range: {scene_port}')
        simulator = ManagedSimulator(name, scene_port, gpu_id,
                                     cfg.output_folder, cfg.simulator_startup_timeout)
        try:
            for row in group:
                try:
                    result = evaluate_episode(cfg, row, simulator, policy, processor, key, device, signature, rank)
                except Exception:
                    atomic_json(Path(cfg.output_folder) / 'errors' / f'rank-{rank}.json',
                                {'episode': episode_key(row), 'traceback': traceback.format_exc()})
                    raise
                completed += 1
                atomic_json(Path(cfg.output_folder) / f'progress-rank-{rank}.json',
                            {'rank': rank, 'completed_this_run': completed, 'last_episode': episode_key(row),
                             'last_steps': result['steps'], 'updated_at': time.time()})
        finally:
            simulator.cleanup()
    print(f'WORKER_COMPLETE rank={rank} episodes={completed}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_folder', '--output_foler', dest='output_folder', required=True)
    parser.add_argument('--eval_dataset', default=RUNTIME_CONFIG['eval_dataset'])
    parser.add_argument('--eval_max_steps', type=int, default=100)
    parser.add_argument('--success_distance', type=float, default=20.0)
    parser.add_argument('--eval_samples_per_env', type=int)
    parser.add_argument('--max_episodes', type=int)
    parser.add_argument('--env_names', nargs='+')
    parser.add_argument('--unnorm_key', default='uavon_raw_json')
    parser.add_argument('--attn_implementation', choices=['eager', 'sdpa', 'flash_attention_2'], default='flash_attention_2')
    parser.add_argument('--airsim_default_port', type=int, default=30400)
    parser.add_argument('--simulator_gpu_ids', help='Comma-separated Vulkan GPU indices; default: match CUDA device UUIDs')
    parser.add_argument('--simulator_startup_timeout', type=float, default=120)
    parser.add_argument('--camera_name', default='uav_on_0')
    parser.add_argument('--settle_seconds', type=float, default=0.1)
    parser.add_argument('--collision_substep_m', type=float, default=0.25)
    parser.add_argument('--collision_yaw_step_degrees', type=float, default=5.0)
    parser.add_argument('--collision_poll_seconds', type=float, default=0.02)
    parser.add_argument('--motion_timeout_seconds', type=float, default=3.0)
    parser.add_argument('--position_tolerance', type=float, default=0.03)
    parser.add_argument('--yaw_tolerance_degrees', type=float, default=1.0)
    parser.add_argument('--log_step_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--summarize_only', action='store_true')
    return parser.parse_args()


def main():
    cfg = parse_args()
    rows = select_episodes(cfg)
    preflight(cfg, rows)
    if cfg.dry_run:
        print(json.dumps({'model_path': cfg.model_path, 'output_folder': cfg.output_folder,
                          'episodes': len(rows), 'maps': len({r['map_name'] for r in rows}),
                          'max_steps': cfg.eval_max_steps, 'unnorm_key': cfg.unnorm_key,
                          'collision_protocol': COLLISION_PROTOCOL,
                          'collision_substep_m': cfg.collision_substep_m,
                          'physics_engine': 'ExternalPhysicsEngine'}, indent=2))
    elif cfg.summarize_only:
        summarize(cfg, rows)
    else:
        run(cfg, rows)


if __name__ == '__main__':
    # torchrun terminates siblings on a worker error; unwind simulator ownership too.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    main()
