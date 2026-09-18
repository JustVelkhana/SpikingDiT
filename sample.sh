#!/usr/bin/env bash
# Generate a few class-conditional images for visualization.
# Usage: bash sample.sh <gpu> <T> <n> <steps> <image_size>
# Example: bash sample.sh 0 4 8 50 256

set -euo pipefail

gpu_no=${1:-0}
T=${2:-4}
num_thresholds=${3:-8}
steps=${4:-50}
size=${5:-256}

cmd="CUDA_VISIBLE_DEVICES=${gpu_no} python main.py \
  --model_name facebook/DiT-XL-2-${size} \
  --pre_rotate \
  --down_online_had \
  --act_steps ${steps} \
  --steps ${steps} \
  --T ${T} \
  --num_thresholds ${num_thresholds} \
  --out_image_dir ./outputs_images/${size}/SNN_T=${T},n=${num_thresholds},mem,step${steps} \
  --cal_data_size 10 \
  --dataset ILSVRC/imagenet-1k \
  --use_spike \
  --use_mem \
  --cfg 1.5"

echo "${cmd}"
eval "${cmd}"
