import utils.hadamard_utils as hadamard_utils
from SNN.spike_layer import snnLinear, snnIdentity
def register_online_had(model):
    for name, module in model.named_modules():
        if isinstance(module, snnLinear):
            had_K, K = hadamard_utils.get_hadK(module.in_features)
            module.online_full_had = True
            module.had_K = had_K
            module.K = K
            module.fp32_had = False
        elif isinstance(module, snnIdentity) and ('softmax' not in name):
            had_K, K = hadamard_utils.get_hadK(module.in_features)
            module.online_full_had = True
            module.had_K = had_K
            module.K = K
            module.fp32_had = False