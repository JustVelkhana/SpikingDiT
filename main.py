import os
import sys
import random
import numpy as np
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from pathlib import Path
from utils import train_utils
from utils.snn_utils import wrap_to_snn_model, init_spiker, get_spike_config
from utils.online_utils import register_online_had
from SNN.change_spike_states import set_spike
from utils.search_utils import get_act_stat
from utils.data_utils import get_loaders
from SNN.spike_neuron import MTH_with_neg_line
from tqdm import tqdm
import torch.multiprocessing as mp
device = torch.device("cuda")


def create_pipeline(args):
  dtype = torch.float32 if args.use_float32 else torch.float16
  
  pipe = DiffusionPipeline.from_pretrained(args.model_name, local_files_only=True, device_map='cuda', torch_dtype={'transformer':dtype, 'default': dtype})
  pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

  wrap_to_snn_model(pipe.transformer)
  pipe.transformer.to(device, dtype=dtype)

  # register on-line hadadamrd transformation
  if args.pre_rotate and args.down_online_had:
    register_online_had(pipe.transformer)


  pipe.transformer.to(device, dtype=dtype)
  dataloader = get_loaders(
    args.calib_dataset,
    data_size=args.cal_data_size,
    seed=args.seed
  )

  pipe.reset_device_map()
  pipe.to('cuda')
  pipe.transformer.to(dtype)

  if args.use_spike:
    activation_stat = get_act_stat(pipe, dataloader, 'max', args.act_steps)

    init_spiker(args, pipe.transformer, activation_stat)

    train_utils.cleanup_memory()

    pipe.reset_device_map()
    pipe.to('cuda')
    pipe.transformer.to(dtype)

    if args.calibration:
      set_spike(pipe.transformer, T=args.T, use_spike=False)
      for name, module in pipe.transformer.named_modules():
        if isinstance(module, MTH_with_neg_line):
            module.calibration = True
            
      for i in range(10):
        test_prompt = [i]
        pipe(test_prompt, num_inference_steps=args.steps, generator=torch.Generator("cuda").manual_seed(17)).images[0]
        
      for name, module in pipe.transformer.named_modules():
        if isinstance(module, MTH_with_neg_line):
            module.calibration = False
            
    if args.use_mem:
      for name, module in pipe.transformer.named_modules():
        if isinstance(module, MTH_with_neg_line):
            module.use_mem = True
          
    set_spike(pipe.transformer, T=args.T, use_spike=True)
  return pipe

def generate_50k_for_eval(args):
    world_size = torch.cuda.device_count()
    mp.spawn(_gpu_process_fn, args=(world_size, args), nprocs=world_size, join=True)
    merge_results(args, world_size)

def _gpu_process_fn(rank, world_size, args):
    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        
        pipe = create_pipeline(args) 
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        n_classes = 1000
        samples_per_class = args.num_samples
        batch_size = args.batch_size
        
        classes_per_gpu = n_classes // world_size
        start_class = rank * classes_per_gpu
        end_class = n_classes if rank == world_size - 1 else (rank + 1) * classes_per_gpu
        
        my_classes = range(start_class, end_class)
        total_images_this_gpu = len(my_classes) * samples_per_class
        
        if '256' in args.model_name:
          local_samples = np.zeros((total_images_this_gpu, 256, 256, 3), dtype=np.uint8)
        else:
          local_samples = np.zeros((total_images_this_gpu, 512, 512, 3), dtype=np.uint8)
        current_idx = 0
        
        iterator = tqdm(my_classes, desc=f"GPU {rank}") if rank == 0 else my_classes

        for class_id in iterator:
            for _ in range(0, samples_per_class, batch_size):
                actual_batch = min(batch_size, samples_per_class - (_))
                
                class_labels = torch.tensor([class_id] * actual_batch, device=device)
                if args.use_mem:
                  for name, module in pipe.transformer.named_modules():
                    if isinstance(module, MTH_with_neg_line):
                      module.mem = None
                with torch.no_grad():
                    output = pipe(
                        class_labels=class_labels,
                        num_inference_steps=args.steps,
                        guidance_scale=args.cfg, #
                        output_type="np"
                    ).images
                    
                    batch_uint8 = (output * 255).astype(np.uint8)
                    
                    local_samples[current_idx : current_idx + actual_batch] = batch_uint8
                    current_idx += actual_batch

        os.makedirs(args.experiment_dir, exist_ok=True)
        save_name = f"temp_samples_rank_{rank}.npz"
        save_path = os.path.join(args.experiment_dir, save_name)
        np.savez(save_path, arr_0=local_samples)
        
        print(f"[GPU {rank}] finish")
        
        del pipe
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[GPU {rank}] error: {e}")
        raise e

def merge_results(args, world_size):
    print("merging results...")
    all_chunks = []
    
    output_dir = Path(args.output_dir)
    logger = train_utils.create_logger(output_dir)
    logger.info(args)
    
    for rank in range(world_size):
        temp_path = os.path.join(args.experiment_dir, f"temp_samples_rank_{rank}.npz")
        if os.path.exists(temp_path):
            data = np.load(temp_path)["arr_0"]
            all_chunks.append(data)
        else:
            logger.info(f"warning! no rank {rank} file!")

    if all_chunks:
        final_images = np.concatenate(all_chunks, axis=0)
        final_path = os.path.join(args.experiment_dir, f"samples_{args.num_samples}k.npz")
        np.savez(final_path, arr_0=final_images)
        logger.info(f"finished! Save to: {final_path}")
        logger.info(f"final shape: {final_images.shape}")
        
        for rank in range(world_size):
            os.remove(os.path.join(args.experiment_dir, f"temp_samples_rank_{rank}.npz"))
    
def main():
  import argparse
  parser = argparse.ArgumentParser()
  # -----------------model setting------------------------------------
  parser.add_argument("--model_name", type=str, help="model name")
  # -----------------snn setting------------------------------------
  parser.add_argument("--T", type=int, default=1, help="time steps of snn")
  parser.add_argument("--num_thresholds", type=int, default=8, help="time steps of snn")
  # -----------------hadamard------------------------------------
  parser.add_argument("--pre_rotate", action="store_true")
  parser.add_argument("--rotate_mode", type=str, default='hadamard')
  parser.add_argument("--down_online_had", action="store_true")
  # -----------------utils------------------------------------
  parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
  parser.add_argument("--calib_dataset", type=str, default='imagenet', help="Calibration data.")
  parser.add_argument("--cal_data_size", type=int, default=50, help="number of calibration data.")
  parser.add_argument("--act_steps", type=int, default=25, help="num_inference_steps of getting activation")
  # -----------------dir------------------------------------
  parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
  parser.add_argument("--dataset", type=str, default="justram/COCO2014-Images") #ILSVRC/imagenet-1k,justram/COCO2014-Images,google-research-datasets/conceptual_12m
  parser.add_argument("--output_dir", default="./log/", type=str, help="direction of logging file")
  parser.add_argument("--save_dir", default=None, type=str, help="direction for saving spiking model")
  parser.add_argument("--out_image_dir", type=str, default="./outputs/imagenet", help="dir for saving output images") 
  parser.add_argument("--experiment_dir", type=str, default="./outputs/DiT_XL_SNN_T4_BaseOptimized_CFG1.5/")
  # -----------------inference setting------------------------------------
  parser.add_argument("--steps", type=int, default=25, help="num_inference_steps")
  parser.add_argument("--cfg", type=float, default=1.5)
  parser.add_argument("--match_cifar_res", action="store_true", help="down sample to 32x32")
  parser.add_argument("--batch_size", type=int, default=25)
  # -----------------eval setting------------------------------------
  parser.add_argument("--num_samples", type=int, default=50, help="sample numbers of eval")
  parser.add_argument("--use_spike", action="store_true")
  parser.add_argument("--generate", action="store_true")
  parser.add_argument("--use_float32", action="store_true")
  # -----------------optimization setting------------------------------------
  parser.add_argument("--calibration", action="store_true")
  parser.add_argument("--use_mem", action="store_true")

  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  args = parser.parse_args()
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

  dtype = torch.float32 if args.use_float32 else torch.float16
  # init logger
  if args.output_dir:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
  if args.cache_dir:
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
  if args.save_dir:
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
  if args.out_image_dir:
    Path(args.out_image_dir).mkdir(parents=True, exist_ok=True)

  output_dir = Path(args.output_dir)
  logger = train_utils.create_logger(output_dir)
  logger.info(args)

  if args.generate:
    generate_50k_for_eval(args)
    exit(0)
  # load fp model

  model = DiffusionPipeline.from_pretrained(args.model_name, local_files_only=True, device_map='cuda', torch_dtype={'transformer':dtype, 'default': dtype})
  model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)

  wrap_to_snn_model(model.transformer)
  model.transformer.to(device, dtype=dtype)

  # register on-line hadadamrd transformation
  if args.pre_rotate and args.down_online_had:
    register_online_had(model.transformer)


  model.transformer.to(device, dtype=dtype)
  dataloader = get_loaders(
    args.calib_dataset,
    data_size=args.cal_data_size,
    seed=args.seed
  )

  model.reset_device_map()
  model.to('cuda')
  model.transformer.to(dtype)

  if args.use_spike:
    activation_stat = get_act_stat(model, dataloader, 'max', args.act_steps)

    logger.info('Insert spike neuron and init threshold.')
    init_spiker(args, model.transformer, activation_stat)

    train_utils.cleanup_memory()

    model.reset_device_map()
    model.to('cuda')
    model.transformer.to(dtype)

    if args.calibration:
      set_spike(model.transformer, T=args.T, use_spike=False)
      for name, module in model.transformer.named_modules():
        if isinstance(module, MTH_with_neg_line):
            module.calibration = True
            
      for i in range(10):
        test_prompt = [i]
        model(test_prompt, num_inference_steps=args.steps, generator=torch.Generator("cuda").manual_seed(17)).images[0]
        
      for name, module in model.transformer.named_modules():
        if isinstance(module, MTH_with_neg_line):
            module.calibration = False
            
    if args.use_mem:
      for name, module in model.transformer.named_modules():
        if isinstance(module, MTH_with_neg_line):
            module.use_mem = True
  
    set_spike(model.transformer, T=args.T, use_spike=True)
  
  if args.save_dir:
    logger.info("start saving model")
    model.transformer.save_pretrained(args.save_dir, safe_serialization=False)
    spike_config = get_spike_config(args)
    train_utils.save_dict_as_json(spike_config, os.path.join(args.save_dir, 'spike_config.json'))
    logger.info(f"save model to {args.save_dir} success")


  for i in range(50):
    if args.use_mem:
      for name, module in model.transformer.named_modules():
        if isinstance(module, MTH_with_neg_line):
          module.mem = None
    test_prompt = [i]
    test_image = model(test_prompt, num_inference_steps=args.steps, generator=torch.Generator("cuda").manual_seed(17)).images[0]
    os.makedirs(args.out_image_dir, exist_ok=True)
    test_image.save(os.path.join(args.out_image_dir, f"SNN_{test_prompt[0]}_T={args.T}.png"))
  



if __name__ == "__main__":
    print(sys.argv)
    main()

