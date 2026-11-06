#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PATH="/home/zzz/anaconda3/envs/openfly/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export USE_TF=0
export OMP_NUM_THREADS=4

usage() {
  cat <<'USAGE'
Usage: bash eval_hiramem.sh [--fine_voxel_size FLOAT] [--coarse_voxel_size FLOAT] [--visit_radius FLOAT]

Each omitted value uses the checkpoint's saved model configuration.
Require 0 < fine_voxel_size < coarse_voxel_size and visit_radius > 0 (finite values).
Overrides are evaluated in a separate output subdirectory named for the parameters.
Set CHECK_ONLY=1 to validate all checkpoints and print effective values without running evaluation.
USAGE
}

memory_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fine_voxel_size|--coarse_voxel_size|--visit_radius)
      if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      memory_args+=("$1" "$2")
      shift 2
      ;;
    --fine_voxel_size=*|--coarse_voxel_size=*|--visit_radius=*)
      memory_args+=("${1%%=*}" "${1#*=}")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Fixed evaluation configuration.
EVAL_DATASET="${EVAL_DATASET:-$(python runtime_config.py --get eval_dataset)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(python runtime_config.py --get output_root)}"
EVAL_MAX_STEPS=100
SUCCESS_DISTANCE=20
ATTN_IMPLEMENTATION="flash_attention_2"
AIRSIM_PORT=30000
NUM_PROCESSES=4
MASTER_PORT=29312
MAX_RESTARTS=0

# Keep parameter sweeps separate from the default run and from each other.
if [[ ${#memory_args[@]} -gt 0 ]]; then
  memory_label="$(python - "${memory_args[@]}" <<'PY'
import math
import sys

overrides = dict(zip(sys.argv[1::2], sys.argv[2::2]))
parts = ['memory']
for name, label in [('fine_voxel_size', 'fine'), ('coarse_voxel_size', 'coarse'), ('visit_radius', 'visit')]:
    raw = overrides.get('--' + name)
    value = 'model'
    if raw is not None:
        try:
            value = float(raw)
        except ValueError:
            raise SystemExit(f'Invalid --{name}: {raw!r}; expected a positive finite number')
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f'Invalid --{name}: {raw!r}; expected a positive finite number')
    parts.extend([label, str(value)])
print('-'.join(parts))
PY
)"
  OUTPUT_ROOT="$OUTPUT_ROOT/$memory_label"
fi

CHECKPOINTS=("${MODEL_PATH:-$(python runtime_config.py --get model_path)}")

# Optional: CHECKPOINT_FILTER=<substring> evaluates only matching checkpoints (e.g. for resuming one failure).
if [[ -n "${CHECKPOINT_FILTER:-}" ]]; then
  filtered=()
  for checkpoint in "${CHECKPOINTS[@]}"; do
    [[ "$checkpoint" == *"$CHECKPOINT_FILTER"* ]] && filtered+=("$checkpoint")
  done
  [[ ${#filtered[@]} -gt 0 ]] || { echo "No checkpoint matches CHECKPOINT_FILTER=$CHECKPOINT_FILTER" >&2; exit 2; }
  CHECKPOINTS=("${filtered[@]}")
fi

checkpoint_label() {
  local checkpoint="$1"
  local run_name checkpoint_name
  run_name="$(basename "$(dirname "$checkpoint")")"
  checkpoint_name="$(basename "$checkpoint")"
  printf '%s--%s' "$run_name" "$checkpoint_name"
}

# Validate every model before starting any simulator.
for checkpoint in "${CHECKPOINTS[@]}"; do
  [[ -d "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
  label="$(checkpoint_label "$checkpoint")"
  python eval_uavon_oct.py \
    --model_path "$checkpoint" \
    --output_folder "$OUTPUT_ROOT/$label" \
    --eval_dataset "$EVAL_DATASET" \
    --eval_max_steps "$EVAL_MAX_STEPS" \
    "${memory_args[@]}" \
    --dry_run
done

# Set CHECK_ONLY=1 for preflight checks; normal execution evaluates all checkpoints.
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  echo "CHECK_ONLY_COMPLETE output=$OUTPUT_ROOT"
  exit 0
fi

mkdir -p "$OUTPUT_ROOT/logs"
timestamp="$(date +%Y%m%d-%H%M%S)"

# Track failed checkpoints for final report
failed_checkpoints=()

for checkpoint in "${CHECKPOINTS[@]}"; do
  label="$(checkpoint_label "$checkpoint")"
  args=(
    --model_path "$checkpoint"
    --output_folder "$OUTPUT_ROOT/$label"
    --eval_dataset "$EVAL_DATASET"
    --eval_max_steps "$EVAL_MAX_STEPS"
    --unnorm_key uavon_raw_json
    --attn_implementation "$ATTN_IMPLEMENTATION"
    --airsim_default_port "$AIRSIM_PORT"
    --success_distance "$SUCCESS_DISTANCE"
    "${memory_args[@]}"
  )
  log="$OUTPUT_ROOT/logs/${label}-${timestamp}.log"
  echo "EVAL_START checkpoint=$checkpoint workers=$NUM_PROCESSES log=$log"
  
  # Run evaluation with error handling
  if torchrun --nnodes=1 --nproc_per_node="$NUM_PROCESSES" \
    --master_addr=127.0.0.1 --master_port="$MASTER_PORT" --max_restarts="$MAX_RESTARTS" \
    eval_uavon_oct.py "${args[@]}" 2>&1 | tee "$log"; then
    echo "EVAL_SUCCESS checkpoint=$checkpoint"
    # Try to run summary, but don't fail if it errors
    python eval_uavon_oct.py "${args[@]}" --summarize_only 2>&1 | tee -a "$log" || {
      echo "SUMMARIZE_FAILED checkpoint=$checkpoint" >&2
    }
  else
    echo "EVAL_FAILED checkpoint=$checkpoint (see $log)" >&2
    failed_checkpoints+=("$label")
  fi
done

# Final report
if [[ ${#failed_checkpoints[@]} -gt 0 ]]; then
  echo "COMPLETED_WITH_FAILURES output=$OUTPUT_ROOT failed=${failed_checkpoints[*]}"
  exit 1
else
  echo "ALL_CHECKPOINTS_COMPLETE output=$OUTPUT_ROOT"
fi
