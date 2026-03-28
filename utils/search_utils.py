from SNN.spike_layer import snnLinear, snnIdentity
from SNN.spike_neuron import MTH_with_neg_line
import functools
from utils import hadamard_utils
import torch
from tqdm import tqdm

def get_act_stat(model, dataloader, accumulate_type='max', steps=1):
    device = model.transformer.device
    act_stat = {}
    
    def stat_tensor(name, tensor, type):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        # ema_factor = 0.99
        if accumulate_type == 'max':
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        elif accumulate_type == 'mean':
            comming_max = torch.mean(tensor, dim=0).float().cpu()
        key_name = f"{name}.{type}"
        if key_name in act_stat:
            act_stat[key_name] = torch.max(act_stat[key_name], comming_max)
            # act_stat[key_name] = ema_factor * act_stat[key_name] + (1-ema_factor) * comming_max
        else:
            act_stat[key_name] = comming_max

    def stat_input_hook(m, x, y, name):
        if m.timestep >= steps*0.4 and m.timestep <= steps*0.6:
            if isinstance(x, tuple):
                x = x[0]
            x_,y_ = x, y
            if m.online_full_had:
                if m.fp32_had:  # Full Hadamard in FP32
                    x_ = hadamard_utils.matmul_hadU_cuda(x_.float(), m.had_K, m.K).to(x_.dtype)
                else:  # Full Hadamard in FP16
                    x_ = hadamard_utils.matmul_hadU_cuda(x_, m.had_K, m.K)
            stat_tensor(name, x_, 'input')
    
    hooks = []
    for name, m in model.transformer.named_modules():
        if isinstance(m, (snnLinear, snnIdentity)):
            m.max_steps = steps
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))
            
    for i in tqdm(range(len(dataloader)), desc='obtain activation stat'):
        data = dataloader[i].unsqueeze(0).to(device)
        model(data, num_inference_steps=steps)

    for h in hooks:
        h.remove()

    return act_stat


def get_cali(model, dataloader, accumulate_type='max', steps=1):
    device = model.transformer.device
    act_stat = {}
    
    def stat_tensor(name, tensor, type):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        # ema_factor = 0.99
        if accumulate_type == 'max':
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        elif accumulate_type == 'mean':
            comming_max = torch.mean(tensor, dim=0).float().cpu()
        key_name = f"{name}.{type}"
        if key_name in act_stat:
            act_stat[key_name] = torch.max(act_stat[key_name], comming_max)
            # act_stat[key_name] = ema_factor * act_stat[key_name] + (1-ema_factor) * comming_max
        else:
            act_stat[key_name] = comming_max

    def stat_input_hook(m, x, y, name):
        if m.timestep >= steps*0.4 and m.timestep <= steps*0.6:
            if isinstance(x, tuple):
                x = x[0]
            x_,y_ = x, y
            if m.online_full_had:
                if m.fp32_had:  # Full Hadamard in FP32
                    x_ = hadamard_utils.matmul_hadU_cuda(x_.float(), m.had_K, m.K).to(x_.dtype)
                else:  # Full Hadamard in FP16
                    x_ = hadamard_utils.matmul_hadU_cuda(x_, m.had_K, m.K)
            stat_tensor(name, x_, 'input')
    
    hooks = []
    for name, m in model.transformer.named_modules():
        if isinstance(m, (snnLinear, snnIdentity)):
            m.max_steps = steps
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))
            
    for i in tqdm(range(len(dataloader)), desc='obtain activation stat'):
        data = dataloader[i].unsqueeze(0).to(device)
        model(data, num_inference_steps=steps)

    for h in hooks:
        h.remove()

    return act_stat
