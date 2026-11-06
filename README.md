# UAV Memory Agent

This project contains **HiRaMem depth** (`model_hiramem_depth`) and its training,
LoRA merging, and UAV-ON evaluation code. Weights, datasets, simulator binaries,
and experiment outputs are stored separately. Shared helpers are included locally;
there is no Python-code dependency on `../train`.

## Reference checkpoint: 23.2% success rate

The recorded evaluation of `model_hiramem_depth-tbptt2-20260928-012757/full-checkpoints-10000`
contains 1,000 episodes, SR 23.2%, OSR 35.7%, and SPL approximately 10.03%.
It uses `eval_uavon_oct.py`, at most 100 actions, a 20 m success threshold,
seed 42, and action statistics `uavon_raw_json`.

Set `model_path` in your local YAML to the downloaded merged checkpoint:

```text
/path/to/himem-agent
```

Only `model_hiramem_depth/` is retained; the no-depth model, Future-3D variants,
and old baseline entry points have been removed. The depth package contains later
optional ablation switches whose standard defaults retain the full model.
Evaluation loads the code saved inside the selected checkpoint, including the
exact saved implementation for the reference checkpoint.

## Layout

| Path | Purpose |
| --- | --- |
| `model_hiramem_depth/` | Depth model, recurrent fine/coarse memory, Q-Former, depth head |
| `data/` | Raw JSON episodes, RGBD transforms, depth labels, memory windows |
| `model/`, `vln_datasets/` | Shared tokenizer, prompt, image-transform, dataset helpers |
| `finetune_acc_mem.py`, `train_mem.sh` | Depth training |
| `merge_lora_v13_dis.py`, `merge_lora.sh` | Merge LoRA and auxiliary weights |
| `eval_uavon_oct.py`, `eval_hiramem.sh`, `eval.sh` | Reference evaluation |
| `configs/runtime.example.yaml` | Published configuration template with `/path/...` placeholders |
| `configs/runtime.yaml` | Local configuration created by the user; excluded from Git |
| `acc_configs/`, `airsim_configs/` | Accelerate and simulator configuration |

## Environment

Activate your environment and enter the project directory:

```bash
conda activate openfly
cd /path/to/uav-memory-agent
```

The project uses PyTorch, Transformers, PEFT, Accelerate, DeepSpeed, timm,
TensorFlow/TFDS/dlimp for the reused data transforms, AirSim, and PyYAML.
FlashAttention 2 is used by the reference scripts. The runtime-path loader itself
needs only Python and PyYAML (`pip install pyyaml`).

## YAML runtime paths

Only `configs/runtime.example.yaml` is distributed. Create your local configuration
before evaluation, then replace its `/path/...` placeholders with real paths:

```bash
cp -n configs/runtime.example.yaml configs/runtime.yaml
```

`cp -n` preserves an existing local configuration. `configs/runtime.yaml` is
excluded by `.gitignore` and must not be included in a release or upload.

`eval_utils.py` and the evaluation launchers read this local YAML.
Select another YAML file with
`UAV_MEMORY_CONFIG` before starting Python or a launcher:

```bash
UAV_MEMORY_CONFIG=/path/to/runtime.yaml bash eval.sh
```

| YAML key | Meaning |
| --- | --- |
| `env_root` | Root containing `TRAIN_ENVS/` and `TEST_ENVS/` |
| `environments` | Dataset scene names mapped to startup scripts relative to `env_root` |
| `airsim_config_dir` | Contains `base_settings_512.json`; normally `../airsim_configs` |
| `eval_dataset` | Evaluation JSON file |
| `model_path` | Merged checkpoint with weights, tokenizer, config, and model code |
| `output_root` | Output directory; normally `../results/depth-23.2` |

Relative top-level paths are resolved against the YAML file's directory.
Scene paths are resolved against `env_root`; absolute scene paths also work.
Path values support `~` and environment-variable expansion.

Set the environment root, evaluation dataset, and merged checkpoint paths for
your machine. Outputs default to this project's `results/depth-23.2/`.
Inspect paths without loading weights or starting a simulator:

```bash
python runtime_config.py
python runtime_config.py --get model_path
```

### Paths for public release

Publish `configs/runtime.example.yaml` only; keep `configs/runtime.yaml` local.
Git ignores the local file. When packaging or uploading through a non-Git tool,
explicitly exclude `configs/runtime.yaml` from the file list as well.

The example uses `/path/to/ENVS`, `/path/to/dataset/merged_episodes_with_action.json`,
and `/path/to/himem-agent`. Users replace these with their paths and retain the
scene mapping. If moving YAML outside `configs/`, also adjust the relative
`airsim_config_dir` and `output_root` values.

Training scripts retain local paths. Before publishing, replace
them with `/path/to/python-env/bin`, `/path/to/base-vla`,
`/path/to/dataset/train.json`, `/path/to/dataset/images`, and
`/path/to/checkpoints` as appropriate. `VLA_PATH` selects pretrained weights;
the current path containing `train/runs-new` is a weight location, not a code import.
Review remaining machine paths before release:

```bash
grep -RnE '/home/zzz|/data1' --include='*.py' --include='*.sh' --include='*.yaml' --include='*.md' .
```

## Evaluation

```bash
CHECK_ONLY=1 bash eval.sh
bash eval.sh
```

The default launcher evaluates only the configured reference checkpoint.
Preflight checks checkpoint assets, dataset, and scene executables without loading
weights or launching a simulator. Results are written under
`<output_root>/<run-name>--<checkpoint-name>/`.

```bash
MODEL_PATH=/path/to/himem-agent EVAL_DATASET=/path/to/eval.json \
  OUTPUT_ROOT=/path/to/results bash eval.sh
```

`bash eval_results.sh` recomputes metrics for the default output directory.
The optional collision-termination protocol is described in
[README_collision.md](README_collision.md); it differs from the 23.2% protocol.

## Training and merging

```bash
bash train_mem.sh
```

The launcher uses 2-step TBPTT, global trajectory batch 16, gradient accumulation 2,
learning rate 2e-5, LoRA rank/alpha 512, fine/coarse voxels 2.5/20 m, and visit radius
20 m, matching the reference run's saved settings. Depth labels and losses are
documented in [model_hiramem_depth/README.md](model_hiramem_depth/README.md).

Override data and base weights:

```bash
VLA_PATH=/path/to/base-vla bash train_mem.sh \
  --raw_train_json /path/to/dataset/train.json \
  --raw_media_root /path/to/dataset/images
```

Merge an adapter into a separate full checkpoint:

```bash
RUN_DIR=/path/to/run CHECKPOINT=checkpoints-10000 bash merge_lora.sh
```

The merger requires memory and depth auxiliary weights and refuses to overwrite
an existing merged checkpoint.

## Training loop

Training uses Accelerate's gradient accumulation and backward/optimizer flow.
The default launcher uses DeepSpeed ZeRO-1; clipping and optimizer updates are
handled by DeepSpeed. Action, depth-pixel, and depth-edge losses retain separate
global denominators across ranks. Recurrent memory is detached at each TBPTT
window boundary. Training metrics, validation, and checkpoint saving are retained.
There is no custom anomaly snapshot, rollback, replay, or episode-quarantine path.
