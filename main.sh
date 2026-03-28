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
