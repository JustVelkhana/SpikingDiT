import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from diffusers.models.attention import _chunked_feed_forward
from diffusers.models.embeddings import get_2d_sincos_pos_embed, SinusoidalPositionalEmbedding
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero
from diffusers.models.attention import Attention, FeedForward
from diffusers.utils.deprecation_utils import deprecate
from utils import hadamard_utils
import math

class snnPatchEmbed(nn.Module):
    def __init__(
        self,
        ori_module,
        T=1
    ):
        super().__init__()

        self.flatten = ori_module.flatten
        self.layer_norm = ori_module.layer_norm
        self.pos_embed_max_size = ori_module.pos_embed_max_size

        self.proj = ori_module.proj
        self.norm = ori_module.norm

        self.patch_size = ori_module.patch_size
        self.height, self.width = ori_module.height, ori_module.width
        self.base_size = ori_module.base_size
        self.interpolation_scale = ori_module.interpolation_scale

        self.pos_embed = ori_module.pos_embed
        self.T = T

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    def forward(self, latent):
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)
        if self.pos_embed is None:
            return latent.to(latent.dtype)
        # Interpolate or crop positional embeddings as needed
        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                    device=latent.device,
                    output_type="pt",
                )
                pos_embed = pos_embed.float().unsqueeze(0)
            else:
                pos_embed = self.pos_embed
        out = (latent + pos_embed).to(latent.dtype)

        
        x = torch.zeros(
            (self.T, *out.shape),
            device=out.device,
            dtype=out.dtype
        )
        x[0] = out
        
        return x
        


class snnAdaLayerNormZero(nn.Module):
    def __init__(self, ori_module, T=1):
        super().__init__()
        self.emb = ori_module.emb

        self.silu = nn.SiLU()
        self.linear = ori_module.linear
        self.norm = snnLayerNorm(ori_module.norm)
        self.T = T

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        
        x = self.norm(x) * (1 + scale_msa[None, :, None, :])
        x = x.clone()
        x[0] = x[0] + shift_msa[None, :, None, :]
        
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# snn modules in transformer blocks (snnAdaLayerNormContinuous is also the norm_out out of transformer blocks)


class snnBasicTransformerBlock(nn.Module):
    def __init__(
        self,
        ori_module,
        T=1
    ):
        super().__init__()
        self.dim = ori_module.dim
        self.num_attention_heads = ori_module.num_attention_heads
        self.attention_head_dim = ori_module.attention_head_dim
        self.dropout = ori_module.dropout
        self.cross_attention_dim = ori_module.cross_attention_dim
        self.activation_fn = ori_module.activation_fn
        self.attention_bias = ori_module.attention_bias
        self.double_self_attention = ori_module.double_self_attention
        self.norm_elementwise_affine = ori_module.norm_elementwise_affine
        self.positional_embeddings = ori_module.positional_embeddings
        self.num_positional_embeddings = ori_module.num_positional_embeddings
        self.only_cross_attention = ori_module.only_cross_attention

        self.use_ada_layer_norm_zero = ori_module.use_ada_layer_norm_zero
        self.use_ada_layer_norm = ori_module.use_ada_layer_norm
        self.use_ada_layer_norm_single = ori_module.use_ada_layer_norm_single
        self.use_layer_norm = ori_module.use_layer_norm
        self.use_ada_layer_norm_continuous = ori_module.use_ada_layer_norm_continuous

        self.norm_type = ori_module.norm_type
        self.num_embeds_ada_norm = ori_module.num_embeds_ada_norm

        self.pos_embed = ori_module.pos_embed

        self.norm1 = ori_module.norm1

        self.attn1 = ori_module.attn1

        self.norm2 = ori_module.norm2

        self.attn2 = ori_module.attn2

        self.norm3 = ori_module.norm3

        self.ff = ori_module.ff

        if hasattr(ori_module, 'fuser'):
            self.fuser = ori_module.fuser

        if hasattr(ori_module, 'scale_shift_table'):
            self.scale_shift_table = ori_module.scale_shift_table

        self._chunk_size = ori_module._chunk_size
        self._chunk_dim = ori_module._chunk_dim

        self.T = T


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        )

        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        attn_output = gate_msa[None, :, None, :] * attn_output


        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm3(hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[None, :, None, :])
        norm_hidden_states = norm_hidden_states.clone()
        norm_hidden_states[0] = norm_hidden_states[0] + shift_mlp[None, :, None, :]

        ff_output = self.ff(norm_hidden_states)

        ff_output = gate_mlp[None, :, None, :] * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states

class snnFeedForward(nn.Module):
    def __init__(
        self,
        ori_module,
        T=1
    ):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net.append(snnGELU(ori_module.net[0])) # act_fn: GELU
        self.net.append(ori_module.net[1]) # Dropout
        self.net.append(snnLinear(ori_module.net[2])) # nn.Linear

        self.T = T

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class snnLayerNorm(nn.Module):
    def __init__(
        self,
        ori_module,
        T=1
    ) -> None:
        super().__init__()
        self.normalized_shape = ori_module.normalized_shape
        self.eps = ori_module.eps
        self.elementwise_affine = ori_module.elementwise_affine
        self.weight = ori_module.weight
        self.bias = ori_module.bias
        self.T = T

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dtype = input.dtype
        input = input.float()
        X = torch.zeros_like(input[0])
        Y_pre = 0
        Out = []
        for t in range(self.T):
            X += input[t]
            Y = F.layer_norm(X, self.normalized_shape, self.weight, self.bias, self.eps)
            Out.append(Y - Y_pre)
            Y_pre = Y
        norm_out = torch.stack(Out, dim=0)
        return norm_out.to(dtype)

class snnLayerNorm_out(nn.Module):
    def __init__(
        self,
        ori_module,
        T=1
    ) -> None:
        super().__init__()
        self.norm_out = ori_module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.norm_out(input.sum(dim=0))
    
def seq_matmul(A, B):
    S_q = A.cumsum(dim=0)
    S_k = B.cumsum(dim=0)
    # term1: S_q @ k^T
    term1 = torch.matmul(S_q, B)
    # term2: q @ S_k^T
    term2 = torch.matmul(A, S_k)
    # term3: q @ k^T
    term3 = torch.matmul(A, B)
    attn_scores = term1 + term2 - term3
    return attn_scores

class snnAttention(nn.Module):
    def __init__(
        self, ori_module, T=1
    ):
        super().__init__()

        self.inner_dim = ori_module.inner_dim
        self.inner_kv_dim = ori_module.inner_kv_dim
        self.query_dim = ori_module.query_dim
        self.use_bias = ori_module.use_bias
        self.is_cross_attention = ori_module.is_cross_attention
        self.cross_attention_dim = ori_module.cross_attention_dim
        self.upcast_attention = ori_module.upcast_attention
        self.upcast_softmax = ori_module.upcast_softmax
        self.rescale_output_factor = ori_module.rescale_output_factor
        self.residual_connection = ori_module.residual_connection
        self.dropout = ori_module.dropout
        self.fused_projections = ori_module.fused_projections
        self.out_dim = ori_module.out_dim
        self.out_context_dim = ori_module.out_context_dim
        self.context_pre_only = ori_module.context_pre_only
        self.pre_only = ori_module.pre_only
        self.is_causal = ori_module.is_causal

        self._from_deprecated_attn_block = ori_module._from_deprecated_attn_block

        self.scale_qk = ori_module.scale_qk
        self.scale = ori_module.scale

        self.heads = ori_module.heads

        self.sliceable_head_dim = ori_module.sliceable_head_dim

        self.added_kv_proj_dim = ori_module.added_kv_proj_dim
        self.only_cross_attention = ori_module.only_cross_attention

        self.group_norm = ori_module.group_norm

        self.spatial_norm = ori_module.spatial_norm

        self.norm_q = ori_module.norm_q
        self.norm_k = ori_module.norm_k

        self.norm_cross = ori_module.norm_cross

        self.to_q = snnLinear(ori_module.to_q)
        self.to_k = snnLinear(ori_module.to_k)
        self.to_v = snnLinear(ori_module.to_v)

        self.add_q_proj = ori_module.add_q_proj
        self.add_k_proj = ori_module.add_k_proj
        self.add_v_proj = ori_module.add_v_proj
        
        self.to_out = nn.ModuleList([])
        self.to_out.append(snnLinear(ori_module.to_out[0]))
        self.to_out.append(ori_module.to_out[1])

        self.norm_added_q = ori_module.norm_added_q
        self.norm_added_k = ori_module.norm_added_k

        self.processor = ori_module.processor

        self.q_Identity = snnIdentity(ori_module.to_q.out_features)
        self.k_Identity = snnIdentity(ori_module.to_q.out_features)
        self.v_Identity = snnIdentity(ori_module.to_q.out_features)
        self.softmax_Identity = snnIdentity(ori_module.to_q.out_features)

        self.T = T
        self.softmax = snnSOFTMAX()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:

        batch_size, sequence_length, _ = (
            hidden_states.shape[1:] if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads


        query = self.q_Identity(query)
        key = self.k_Identity(key)
        value = self.v_Identity(value)


        query = query.view(self.T, batch_size, -1, self.heads, head_dim).transpose(2, 3)

        key = key.view(self.T, batch_size, -1, self.heads, head_dim).transpose(2, 3)
        value = value.view(self.T, batch_size, -1, self.heads, head_dim).transpose(2, 3)

        scores = seq_matmul(query, key.transpose(-2, -1) / math.sqrt(query.size(-1)))

        attn_weights = self.softmax(scores, self.T)
        
        attn_weights = self.softmax_Identity(attn_weights)
        
        hidden_states = seq_matmul(attn_weights, value)
        
        hidden_states = hidden_states.transpose(2, 3).reshape(self.T, batch_size, -1, self.heads * head_dim)
        
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states


# --------------------------------------------------------------------------------------------------------------

class snnLinear(nn.Module):
    def __init__(
            self,
            ori_module,
            T=1
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight', ori_module.weight)  # trainable
        if ori_module.bias is not None:
            self.register_buffer('bias', ori_module.bias)
        else:
            self.bias = None
        self.in_features = ori_module.in_features
        self.out_features = ori_module.out_features
        self.online_full_had = False
        self.fp32_had = False
        self.use_temporary_parameter = False

        self.use_spike = False
        self.T = T

        self.transpose_had = False
        
        self.timestep = 0
        self.max_steps = 1
        
    def forward(self, input: torch.Tensor):
        self.timestep = (self.timestep % self.max_steps) + 1
        
        input_dtype = input.dtype

        if self.online_full_had:
            if self.fp32_had:  # Full Hadamard in FP32
                input = hadamard_utils.matmul_hadU_cuda(input.float(), self.had_K, self.K).to(input_dtype)
            else:  # Full Hadamard in FP16
                input = hadamard_utils.matmul_hadU_cuda(input, self.had_K, self.K)

        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight


        if self.use_spike:
            input = self.input_quantizer(input)
        elif hasattr(self, 'input_quantizer') and self.input_quantizer.calibration and (self.timestep >= self.max_steps*0.4 and self.timestep <= self.max_steps*0.6):
            self.input_quantizer(input)

        if self.online_full_had:
            if self.fp32_had:  # Full Hadamard in FP32
                input = hadamard_utils.matmul_hadU_cuda(input.float(), self.had_K.T, self.K).to(input_dtype)
            else:  # Full Hadamard in FP16
                input = hadamard_utils.matmul_hadU_cuda(input, self.had_K.T, self.K)

        out = self.fwd_func(input, weight, None, **self.fwd_kwargs)
        if self.bias is not None:
            b = self.bias.view(*([1] * (out[0].dim() - 1)), -1)
            out = out.clone()
            out[0] = out[0] + b
            
        return out


class snnIdentity(nn.Module):
    def __init__(
            self,
            in_features,
    ):
        super().__init__()
        self.in_features = in_features
        self.use_spike = False
        self.online_full_had = False

        self.fp32_had = False
        
        self.timestep = 0
        self.max_steps = 1
        
    def forward(self, x):
        self.timestep = (self.timestep % self.max_steps) + 1
        
        input_dtype = x.dtype

        if self.online_full_had:
            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(input_dtype)
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        if self.use_spike:
            x = self.input_quantizer(x)
        elif hasattr(self, 'input_quantizer') and self.input_quantizer.calibration and (self.timestep >= self.max_steps*0.4 and self.timestep <= self.max_steps*0.6):
            self.input_quantizer(x)
            
        if self.online_full_had:
            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K.T, self.K).to(input_dtype)
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K.T, self.K)
        return x


# --------------------------------------------------------------------------------------------------------------


class snnGELU(nn.Module):
    def __init__(self, ori_module, T=1):
        super().__init__()
        self.proj = snnLinear(ori_module.proj)
        self.approximate = ori_module.approximate

        self.T = T

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate, approximate=self.approximate)
    
    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        X = torch.zeros_like(hidden_states[0])
        Y_pre = 0
        Out = []
        for t in range(self.T):
            X = X + hidden_states[t]
            Y = self.gelu(X)
            Out.append(Y - Y_pre)
            Y_pre = Y
        hidden_states = torch.stack(Out, dim=0)
        return hidden_states.to(dtype)

class snnSILU(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = nn.SiLU()

    def forward(self, hidden_states, T):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        X = torch.zeros_like(hidden_states[0])
        Y_pre = 0
        Out = []
        for t in range(T):
            X += hidden_states[t]
            Y = self.silu(X)
            Out.append(Y - Y_pre)
            Y_pre = Y
        hidden_states = torch.stack(Out, dim=0)

        return hidden_states.to(dtype)
    
class snnSOFTMAX(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, T):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        X = torch.zeros_like(hidden_states[0])
        Y_pre = 0
        Out = []
        for t in range(T):
            X += hidden_states[t]
            Y = F.softmax(X, dim=-1)
            Out.append(Y - Y_pre)
            Y_pre = Y
        hidden_states = torch.stack(Out, dim=0)
        return hidden_states.to(dtype)
