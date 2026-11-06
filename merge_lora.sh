#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/home/zzz/anaconda3/envs/openfly/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export USE_TF=0

RUN_DIR="${RUN_DIR:-/home/zzz/code/UAV_ON/oct_agent/openvla-runs/openvla-uavon/hiramem/model_hiramem_depth-tbptt2-20260928-012757}"
# CHECKPOINT selects one checkpoint; CHECKPOINTS accepts a space-separated batch.
read -r -a checkpoints <<< "${CHECKPOINT:-${CHECKPOINTS:-checkpoints-12500 checkpoints-10000 checkpoints-7500}}"
base_args=()
if [[ -n "${BASE_VLA:-}" ]]; then
  base_args=(--vla_path "$BASE_VLA")
fi
for checkpoint in "${checkpoints[@]}"; do
  python merge_lora_v13_dis.py \
    --run_dir "$RUN_DIR" \
    --merge_lora_path "$checkpoint" \
    --init_oct_model True \
    --device "${MERGE_DEVICE:-auto}" \
    "${base_args[@]}" "$@"
done
