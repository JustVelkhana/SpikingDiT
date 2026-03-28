from datasets import load_dataset
import torch
def get_loaders(
    name, data_size=128, seed=0
):
    if 'imagenet' in name:
        return get_imagenet(data_size, seed)
    else:
        raise NotImplementedError


def get_imagenet(data_size, seed):
    print("get_imagenet")
    dataset = load_dataset("ILSVRC/imagenet-1k", split='validation', streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    dataset = dataset.take(data_size)
    # labels = dataset.features['label'].int2str
    unique_labels = {example["label"] for example in dataset}
    prompts = sorted(unique_labels)
    prompts = torch.tensor(prompts, device="cuda")
    return prompts