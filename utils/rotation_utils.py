import torch
import utils.model_utils as model_utils
from utils.hadamard_utils import random_hadamard_matrix
from utils.train_utils import cleanup_memory
import tqdm


DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


@torch.inference_mode()
def rotate_model(model, rotate_mode='hadamard', online=False, config=None):
    Q = get_orthogonal_matrix(model.config.joint_attention_dim,rotate_mode,DEV)
    num_attention_heads = config.num_attention_heads
    model_dim = config.joint_attention_dim
    head_dim = model_dim // num_attention_heads
    Q_ov = get_orthogonal_matrix(head_dim,rotate_mode,DEV)


    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    cleanup_memory()
    layers = model_utils.get_transformer_layers(model, model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type, online)
        rotate_ov_proj(layers[idx], Q_ov, model_type, num_attention_heads, head_dim)


def get_orthogonal_matrix(size, mode, device=DEV):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')
    
def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for W in model_utils.get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype) 


def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model, model_type=model_utils.model_type_extractor(model))
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.

    for W in [layer.attn.to_q, layer.attn.to_k, layer.attn.to_v]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    if layer.use_dual_attention:
        for W in [layer.attn2.to_q, layer.attn2.to_k, layer.attn2.to_v]:
            dtype = W.weight.dtype
            W_ = W.weight.to(device=DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == model_utils.STABLE_DIFFUSION_TRANSFORMER:
        W = layer.attn.to_out
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

    if layer.use_dual_attention:
        if model_type == model_utils.STABLE_DIFFUSION_TRANSFORMER:
            W = layer.attn2.to_out
        else:
            raise ValueError(f'Unknown model type {model_type}')

        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device=DEV, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_input(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == model_utils.STABLE_DIFFUSION_TRANSFORMER:
        mlp_inputs = [layer.ff.net[0].proj, layer.ff_context.net[0].proj]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    
def rotate_mlp_output(layer, Q, model_type, online):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.STABLE_DIFFUSION_TRANSFORMER:
        W = [layer.ff.net[2], layer.ff_context.net[2]]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if online:
        apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output (require on-line hadamard)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, Q, model_type, num_attention_heads, head_dim):
    v_proj = layer.attn.to_v
    if model_type == STABLE_DIFFUSION_TRANSFORMER:
        o_proj = layer.attn.to_out
    else:
        raise ValueError(f'Unknown model type {model_type}')
    # v_proj output
    dtype = v_proj.weight.data.dtype
    W_ = v_proj.weight.data.to(device=DEV, dtype=torch.float64)
    shape = W_.shape
    W_ = W_.reshape(1,shape[0],-1)
    v_proj.weight.data = torch.matmul(Q.T.unsqueeze(0), W_).reshape(shape).to(device='cpu', dtype=dtype)
    if v_proj.bias is not None:
        B_ = v_proj.bias.data.to(device=DEV, dtype=torch.float64)
        shape = B_.shape
        B_ = B_.reshape(1,shape[0], -1)
        v_proj.bias.data = torch.matmul(Q.T, B_).flatten().to(device="cpu", dtype=dtype)
    
    # o_proj input
    dtype = o_proj.weight.data.dtype
    W_ = o_proj.weight.data.to(device=DEV, dtype=torch.float64)
    shape = W_.shape
    W_ = W_.reshape(-1, num_attention_heads, shape[-1]//num_attention_heads,) 
    o_proj.weight.data = torch.matmul(W_, Q).reshape(shape).to(device='cpu', dtype=dtype)