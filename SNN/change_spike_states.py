from SNN.spike_layer import snnLinear, snnIdentity


def set_spike(model, T=1, use_spike=False):
    for name, module in model.named_modules():
        if isinstance(module, (snnIdentity, snnLinear)) and hasattr(module, 'input_quantizer'):
            module.use_spike = use_spike
        if hasattr(module, 'T'):
            module.T = T