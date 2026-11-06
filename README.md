# UAV Memory Agent

HiRaMem depth (`himem_agent`) model code, training, LoRA merging, and
UAV-ON evaluation. There are three executable workflow entry points:
`train.py`, `merge.py`, and `eval.py`. Shared code is included in this repository.
Weights, datasets, simulator binaries, and experiment outputs are stored separately.

## Reference result

The recorded evaluation of
`model_hiramem_depth-tbptt2-20260928-012757/full-checkpoints-10000` contains
1,000 episodes, SR **23.2%**, OSR **35.7%**, and SPL **10.03%**.
Its original evaluator, `eval_uavon_oct.py`, is now consolidated into `eval.py`.
The protocol uses at most 100 actions, a 20 m success threshold, seed 42, and
action statistics `uavon_raw_json`. Pose updates ignore collisions; collisions
do not terminate episodes. SPL uses action counts, as in the recorded result.

New training checkpoints save their model code under `himem_agent/`.
Evaluation and merging still load the package recorded in each checkpoint, so
existing `model_hiramem_depth` checkpoints remain usable without moving their files.
Warm-start training accepts the old package name and saves new checkpoints as `himem_agent`.
The source package retains optional memory/depth ablation switches; their default
values enable the full depth model.

## Layout

| Path | Purpose |
| --- | --- |
| `train.py` | Training configuration, distributed launch, and training loop |
| `merge.py` | Merge LoRA and memory/depth auxiliary weights |
| `eval.py` | Distributed evaluation, resume, and metric summary |
| `himem_agent/` | Recurrent fine/coarse memory, Q-Former, depth head |
| `data/` | Raw JSON episodes, input transforms, batch collation, depth labels, memory windows |
| `evaluation/` | Simulator helpers and reference metrics |
| `model/` | Shared tokenizer, prompt, image helpers, model registry, and loss utilities |
| `configs/runtime.example.yaml` | Runtime path template |
| `acc_configs/`, `airsim_configs/` | Accelerate and simulator configuration |

## Environment

```bash
conda activate openfly
cd /path/to/uav-memory-agent
```

The code uses PyTorch, Transformers, PEFT, Accelerate, DeepSpeed, timm,
AirSim, and PyYAML. Input transforms and collation live in `data/`.
The default model attention implementation is FlashAttention 2.

## Runtime configuration

Only `configs/runtime.example.yaml` is published. Create your local configuration
and replace its `/path/...` placeholders:

```bash
cp -n configs/runtime.example.yaml configs/runtime.yaml
```

`configs/runtime.yaml` is ignored by Git. Also exclude it when uploading or
packaging files outside Git. `cp -n` preserves any existing local configuration.

| YAML key | Meaning |
| --- | --- |
| `env_root` | Root containing `TRAIN_ENVS/` and `TEST_ENVS/` |
| `environments` | Dataset scene names mapped to startup scripts relative to `env_root` |
| `airsim_config_dir` | Directory containing `base_settings_512.json` |
| `eval_dataset` | Evaluation JSON file |
| `model_path` | Merged checkpoint with weights, tokenizer, configuration, and model code |
| `output_root` | Evaluation output root |

Relative top-level paths resolve against the YAML directory; scene paths resolve
against `env_root`. Absolute paths, `~`, and environment variables are supported.
Select another YAML with `UAV_MEMORY_CONFIG=/path/to/runtime.yaml` before running
`python eval.py`. If moving YAML outside `configs/`, adjust relative paths too.

## Evaluation

```bash
# Check checkpoint assets, dataset, and scene executables.
python eval.py --dry_run

# Launch four workers and summarize after all workers finish.
python eval.py
```

The default output is `<output_root>/<run-name>--<checkpoint-name>/`.
Override paths or worker count with command-line arguments:

```bash
CUDA_VISIBLE_DEVICES=0,1 python eval.py --num_processes 2 \
  --model_path /path/to/himem-agent \
  --eval_dataset /path/to/eval.json \
  --output_folder /path/to/results
```

Use `--num_processes 1` for a single worker. `--master_port` defaults to 29312.
Rerunning the same configuration resumes completed episodes. To recompute metrics
without running episodes, use the same dataset/model options and output folder:

```bash
python eval.py --summarize_only --output_folder /path/to/results
```

## Training

```bash
python train.py --dry_run True
python train.py
```

The script launches Accelerate with DeepSpeed ZeRO-1 and four processes by default.
The default settings are 2-step TBPTT, global trajectory batch 16, gradient
accumulation 2, learning rate 2e-5, LoRA rank/alpha 512, fine/coarse voxel sizes
2.5/20 m, and visit radius 20 m. Each run gets a timestamped name.

Training retains local data and base-model path defaults. Set your own paths:

```bash
python train.py --num_processes 4 \
  --vla_path /path/to/base-vla \
  --raw_train_json /path/to/dataset/train.json \
  --raw_media_root /path/to/dataset/images \
  --run_root_dir /path/to/checkpoints
```

`--master_port` defaults to 29331. `--deepspeed_config` selects another Accelerate
configuration. `--run_id_note` sets an explicit run name. Run `python train.py --help`
for all training options. The base-model path is a weight location; the repository
does not import code from an external `train` directory.

Training uses Accelerate accumulation/backward and optimizer updates. Action,
depth-pixel, and depth-edge losses use separate global denominators across ranks.
Memory is detached at each TBPTT window boundary. Training metrics, validation,
and checkpoint saving are included. See
[the depth model documentation](himem_agent/README.md) for depth targets
and model options.

## Merging

```bash
python merge.py --run_dir /path/to/run --checkpoint checkpoints-10000 --dry_run
python merge.py --run_dir /path/to/run --checkpoint checkpoints-10000
```

`--checkpoint` defaults to `checkpoints-10000`. The base-model path is read from
the checkpoint's training configuration; override it with `--vla_path`.
The output is `<run_dir>/full-<checkpoint-name>/`. The merger requires memory and
depth auxiliary weights, checks the merged model, and refuses to overwrite an
existing output. Set the resulting directory as `model_path` in your local YAML.
