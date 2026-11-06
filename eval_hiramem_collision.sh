#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PATH="/home/zzz/anaconda3/envs/openfly/bin:$PATH"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export USE_TF=0
export OMP_NUM_THREADS=4

# Fixed evaluation configuration.
EVAL_DATASET="${EVAL_DATASET:-$(python runtime_config.py --get eval_dataset)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(python runtime_config.py --get output_root)/collision}"
EVAL_MAX_STEPS=100
SUCCESS_DISTANCE=20
ATTN_IMPLEMENTATION="flash_attention_2"
AIRSIM_PORT=30400
NUM_PROCESSES=4
MASTER_PORT=29332
MAX_RESTARTS=0

CHECKPOINTS=("${MODEL_PATH:-$(python runtime_config.py --get model_path)}")

checkpoint_label() {
  local checkpoint="$1"
  local run_name checkpoint_name
  run_name="$(basename "$(dirname "$checkpoint")")"
  checkpoint_name="$(basename "$checkpoint")"
  printf '%s--%s' "$run_name" "$checkpoint_name"
}

mkdir -p "$OUTPUT_ROOT/logs"
timestamp="$(date +%Y%m%d-%H%M%S)"

# Validate every model before starting any simulator.
for checkpoint in "${CHECKPOINTS[@]}"; do
  [[ -d "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
  label="$(checkpoint_label "$checkpoint")"
  python eval_uavon_oct_collision.py \
    --model_path "$checkpoint" \
    --output_folder "$OUTPUT_ROOT/$label" \
    --eval_dataset "$EVAL_DATASET" \
    --eval_max_steps "$EVAL_MAX_STEPS" \
    --dry_run
done

# Set CHECK_ONLY=1 for a path-only check; normal execution evaluates all listed models.
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  echo "CHECK_ONLY_COMPLETE output=$OUTPUT_ROOT"
  exit 0
fi

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
  )
  log="$OUTPUT_ROOT/logs/${label}-${timestamp}.log"
  echo "EVAL_START checkpoint=$checkpoint workers=$NUM_PROCESSES log=$log"
  torchrun --nnodes=1 --nproc_per_node="$NUM_PROCESSES" \
    --master_addr=127.0.0.1 --master_port="$MASTER_PORT" --max_restarts="$MAX_RESTARTS" \
    eval_uavon_oct_collision.py "${args[@]}" 2>&1 | tee "$log"
  python eval_uavon_oct_collision.py "${args[@]}" --summarize_only 2>&1 | tee -a "$log"
done

echo "ALL_CHECKPOINTS_COMPLETE output=$OUTPUT_ROOT"
