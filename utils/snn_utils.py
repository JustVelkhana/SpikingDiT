from SNN.spike_layer import snnBasicTransformerBlock, snnFeedForward, snnLayerNorm, snnAttention
from SNN.spike_layer import snnPatchEmbed, snnAdaLayerNormZero, snnLayerNorm_out
from SNN.spike_layer import snnLinear, snnIdentity
from SNN.spike_neuron import MTH_with_neg_line
from diffusers.models.attention import Attention, FeedForward, BasicTransformerBlock
from diffusers.models.embeddings import PatchEmbed
from diffusers.models.normalization import AdaLayerNormZero
import torch.nn as nn
import torch

def wrap_to_snn_model(model):
    '''
    replace stable diffusion modules to snn modules, init T=1
    '''
    for name, module in model.named_modules():
        if 'pos_embed' in name and isinstance(module, PatchEmbed):
            snnEmbed1 = snnPatchEmbed(module)
            set_op_by_name(model, name, snnEmbed1)
            del module
        elif 'norm1' in name and isinstance(module, AdaLayerNormZero):
            snnNorm = snnAdaLayerNormZero(module)
            set_op_by_name(model, name, snnNorm)
            del module
        elif 'block' in name and isinstance(module, BasicTransformerBlock):
            snnblock = snnBasicTransformerBlock(module)
            set_op_by_name(model, name, snnblock)
            del module
        elif 'attn1' in name and isinstance(module, Attention):
            snnAttn = snnAttention(module)
            set_op_by_name(model, name, snnAttn)
            del module
        elif 'norm3' in name and isinstance(module, nn.LayerNorm):
            snnNorm = snnLayerNorm(module)
            set_op_by_name(model, name, snnNorm)
            del module
        elif 'ff' in name and isinstance(module, FeedForward):
            snnff = snnFeedForward(module)
            set_op_by_name(model, name, snnff)
            del module
        elif 'norm_out' in name and isinstance(module, nn.LayerNorm):
            snnNorm = snnLayerNorm_out(module)
            set_op_by_name(model, name, snnNorm)
            del module


def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)



def init_spiker(args, model, activation_stat=None, minmax_init=True):
    input_group_size = 32
    for name, module in model.named_modules():
        if isinstance(module, (snnIdentity, snnLinear)) and ('softmax' not in name):
            input_stat = activation_stat[f'{name}.input'] if activation_stat is not None else None
            shape = (1, module.in_features)
            module.input_quantizer = MTH_with_neg_line(shape, input_group_size, 
                                                            item_stat=input_stat,
                                                            minmax_init=minmax_init,
                                                            T=args.T,
                                                            num_thresholds=args.num_thresholds)
        elif isinstance(module, (snnIdentity)) and ('softmax' in name):
            input_stat = activation_stat[f'{name}.input'] if activation_stat is not None else None
            if "256" in args.model_name:
                shape = (1, 256)
            else:
                shape = (1, 1024)
            module.input_quantizer = MTH_with_neg_line(shape, input_group_size, 
                                                            item_stat=input_stat,
                                                            minmax_init=minmax_init,
                                                            T=args.T,
                                                            num_thresholds=args.num_thresholds)


def get_spike_config(args):
    spike_config = {}
    spike_config["down_online_had"] = args.down_online_had and args.pre_rotate
    spike_config["T"] = args.T
    return spike_config