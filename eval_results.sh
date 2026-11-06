#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/home/zzz/anaconda3/envs/openfly/bin:$PATH"

# Keep the default output folder aligned with eval.sh.
MODEL_PATH="${MODEL_PATH:-$(python runtime_config.py --get model_path)}"
LABEL="$(basename "$(dirname "$MODEL_PATH")")--$(basename "$MODEL_PATH")"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-$(python runtime_config.py --get output_root)/$LABEL}"
DATASET_PATH="${DATASET_PATH:-$(python runtime_config.py --get eval_dataset)}"

args=(--output_folder "$OUTPUT_FOLDER" --dataset_path "$DATASET_PATH")
# Without SAMPLES_PATH, aggregate the per-episode JSON files in OUTPUT_FOLDER/temp.
if [[ -n "${SAMPLES_PATH:-}" ]]; then
  args+=(--samples_path "$SAMPLES_PATH")
fi

exec python eval_results_v2.py "${args[@]}" "$@"
