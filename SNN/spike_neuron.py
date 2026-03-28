import torch
import torch.nn as nn

class MTH_with_neg_line(nn.Module):
    def __init__(
            self,
            shape,
            group_size=-1,
            item_stat=None,
            minmax_init=True,
            T=1,
            num_thresholds=8):
        super(MTH_with_neg_line, self).__init__()
        self.shape = shape
        self.group_size = group_size if group_size != -1 else shape[-1]
        assert shape[-1] % group_size == 0
        self.inc_groups = shape[-1] // self.group_size
        self.num_thresholds = num_thresholds
        self.T = T
        self.scale = nn.Parameter(torch.ones((self.inc_groups, 1)), requires_grad=False)
        self.find_activation_param(item_stat, minmax_init)
        
        self.base = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.calibration = False
        self.use_mem = False
        self.mem = None
        
    def find_activation_param(self, item_stat, minmax_init):
        if minmax_init and item_stat is not None:
            x = item_stat.reshape(-1, self.group_size)
            xmax = x.amax([-1], keepdim=True)
            scale = 2 * xmax/self.T
            self.scale.data.copy_(scale)

    def _spike(self, x_in, base_val):
        device = x_in.device
        dtype = x_in.dtype
        
        t_steps = base_val ** torch.arange(self.num_thresholds, device=device, dtype=dtype)
        t_shape = [self.num_thresholds] + [1] * (x_in.dim() + 1)
        t_steps = t_steps.view(*t_shape)
        
        thre_shaped = self.scale.unsqueeze(0) / t_steps  # [num_thresholds, ...]
        thre_threshold = thre_shaped * 0.75
        
        x_reshaped = x_in.reshape(*x_in.shape[:-1], -1, self.group_size)
        
        if self.use_mem:
            mem = torch.zeros_like(x_reshaped[0]) if self.mem is None else self.mem
        else:
            mem = torch.zeros_like(x_reshaped[0])
        
        spike_all = torch.empty((self.T, *mem.shape), device=device, dtype=dtype)
        
        for t in range(self.T):
            mem = mem + x_reshaped[t]
            sign = torch.sign(mem)
            abs_mem = torch.abs(mem)
            
            indicators = (abs_mem.unsqueeze(0) >= thre_threshold).to(dtype)
            
            pos_true = indicators.clone()
            if self.num_thresholds > 1:
                pos_true[1:] = indicators[1:] - indicators[:-1]
                
            spike = torch.sum(thre_shaped * (sign * pos_true), dim=0).squeeze(0)
            
            mem = mem - spike
            spike_all[t] = spike
        if self.use_mem:
            self.mem = mem
        out = spike_all.view(self.T, *x_in.shape[1:-1], -1)
        return out

    
    def optimize_base(self, x):
        best_mse = float('inf')
        best_base = self.base.item()
        search_range = torch.linspace(1.0, 4, steps=10).to(x.device)
        for candidate in search_range:
            with torch.no_grad():
                out = self._spike(x, candidate)
                mse = torch.mean((x.sum(dim=0) - out.sum(dim=0)) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_base = candidate.item()
        self.base.data.fill_(best_base)
        
    def forward(self, x):
        if self.calibration:
            self.optimize_base(x)
        return self._spike(x, self.base)
