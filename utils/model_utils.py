import diffusers

STABLE_DIFFUSION_TRANSFORMER = diffusers.models.SD3Transformer2DModel

def get_model_type(model):
    if isinstance(model, STABLE_DIFFUSION_TRANSFORMER):
        return STABLE_DIFFUSION_TRANSFORMER
    else:
        raise ValueError(f'Unknown model type {model}')
    

# def get_embeddings(model, model_type) -> list[torch.nn.Module]:
def get_embeddings(model, model_type):
    if model_type == STABLE_DIFFUSION_TRANSFORMER:
        return [model.pos_embed.proj, model.time_text_embed.timestep_embedder, model.context_embedder]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    

def get_transformer_layers(model, model_type):
    if model_type == STABLE_DIFFUSION_TRANSFORMER:
        return [layer for layer in model.transformer_blocks]
    else:
        raise ValueError(f'Unknown model type {model_type}')

def model_type_extractor(model):
    if isinstance(model, STABLE_DIFFUSION_TRANSFORMER):
        return STABLE_DIFFUSION_TRANSFORMER
    else:
        raise ValueError(f'Unknown model type {model}')
    

def get_rope_function_name(model):
    if isinstance(model, STABLE_DIFFUSION_TRANSFORMER):
        return "apply_rotary_pos_emb"
    raise NotImplementedError