#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PATH="/home/zzz/anaconda3/envs/openfly/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
MODEL_MODULE="${MODEL_MODULE:-model_hiramem_depth}"

accelerate launch \
  --config_file "${HIRAMEM_DS_CONFIG:-acc_configs/deep_z1_config.yaml}" \
  --num_processes="${NUM_PROCESSES:-4}" \
  --main_process_port="${MASTER_PORT:-29331}" \
  finetune_acc_mem.py \
  --model_module "$MODEL_MODULE" \
  --model_class "${MODEL_CLASS:-OpenVLAForActionPrediction}" \
  --depth_loss_weight "${DEPTH_LOSS_WEIGHT:-0.1}" \
  --depth_edge_weight "${DEPTH_EDGE_WEIGHT:-0.1}" \
  --depth_min "${DEPTH_MIN:-0.1}" \
  --depth_max "${DEPTH_MAX:-100.0}" \
  --depth_near_distance "${DEPTH_NEAR_DISTANCE:-10.0}" \
  --depth_near_weight "${DEPTH_NEAR_WEIGHT:-3.0}" \
  --depth_huber_delta "${DEPTH_HUBER_DELTA:-1.0}" \
  --vla_path "${VLA_PATH:-/home/zzz/code/OpenFly-Platform/train/runs-new/openvla-7b+vln_mix+b4+lr-1e-05-ep5+lora-r256-a512+dropout-0.0--image_aug/full-checkpoints-62077}" \
  --tbptt_steps "${TBPTT_STEPS:-2}" \
  --total_batch_size 16 \
  --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS:-2}" \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --epoches 4 \
  --lora_rank 512 \
  --lora_alpha 512 \
  --fine_voxel_size 2.5 \
  --coarse_voxel_size 20.0 \
  --visit_radius 20.0 \
  --num_query_tokens "${NUM_QUERY_TOKENS:-128}" \
  --num_frontier_tokens "${NUM_FRONTIER_TOKENS:-16}" \
  --gradient_checkpointing True \
  --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --run_id_note "${RUN_ID:-${MODEL_MODULE}-tbptt${TBPTT_STEPS:-2}-$(date +%Y%m%d-%H%M%S)}" \
  "$@"
