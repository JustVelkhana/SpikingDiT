#!/usr/bin/env bash
# Generate 10k ImageNet samples for FID / sFID / IS evaluation.
# Usage: bash main.sh <gpu> <T> <n> <steps> <image_size>
# Example: bash main.sh 0 4 8 50 256
#
# Recommended paper settings:
#   T=4, n=6/8/10, steps=25/50/100, image_size=256/512

set -euo pipefail

gpu_no=$1
T=$2
num_thresholds=$3
steps=$4
size=$5
de=${de:-""}

cmd="CUDA_VISIBLE_DEVICES=$gpu_no python"
if [[ "$de" == "de" ]]; then
    cmd="$cmd -m debugpy --listen 6868 --wait-for-client"
fi

cmd="$cmd main.py \
--model_name facebook/DiT-XL-2-$size \
--pre_rotate \
--down_online_had \
--act_steps $steps \
--steps $steps \
--T $T \
--generate \
--batch_size 50 \
--num_samples 10 \
--num_thresholds $num_thresholds \
--out_image_dir ./outputs_images/$size/SNN_T=$T,n=$num_thresholds,mem,step$steps \
--cal_data_size 10 \
--dataset ILSVRC/imagenet-1k \
--experiment_dir ./outputs/$size/SNN_T=$T,n=$num_thresholds,mem,step$steps \
--use_spike \
--use_mem \
--cfg 1.5"
echo $cmd
eval $cmd
