import os
import random
from typing import Literal

import numpy
import torch

from src.module import MODEL_PRESETS
from src.module.configuration import QiChatConfig
from src.module.model import QiChatForCausalLM
from src.utils.path import data_path


def set_seed(seed: int = 36):
    """
    Set the random seed for reproducibility.
    :param seed: The seed value to set.
    """
    random.seed(seed)
    numpy.random.seed(seed=seed)
    torch.manual_seed(seed=seed)


def get_device(device: Literal['auto', 'cuda', 'mps', 'cpu']) -> torch.device:
    """
    Get the device to run the model on.
    Args:
        device: Device to use, e.g., 'cpu', 'cuda', 'mps', or 'auto' to auto-detect.
    Returns:
        torch.device
    """
    if device == 'auto':
        if torch.cuda.is_available():
            use_device = torch.device('cuda')
        elif torch.mps.is_available():
            use_device = torch.device("mps")
        else:
            use_device = torch.device("cpu")
    else:
        if device == "cuda" and torch.cuda.is_available():
            use_device = torch.device("cuda")
        elif device == "mps" and torch.mps.is_available():
            use_device = torch.device("mps")
        else:
            if device != "cpu":
                print(f'Warning: Device {device} is not available. Falling back to CPU.')
            use_device = torch.device("cpu")
    return use_device


def init_model(
    param_size: Literal['Tiny', '73M', '0.3B', '0.6B', '1.3B', '3.2B'] = 'Tiny',
    attn_implementation: Literal['eager', 'memory_efficient', 'sdpa', 'flash_attention_2', 'flash_attention_3'] = 'flash_attention_2',
    dtype: Literal['float16', 'bfloat16', 'float32'] | torch.dtype = 'bfloat16',
    additional_model_config_kwargs: dict | None = None,
    checkpoint: dict | None = None,
    use_device: Literal['auto', 'cuda', 'mps', 'cpu'] | torch.device | None = None,
) -> QiChatForCausalLM:
    """
    Initialize the QiChat model for inference.
    Args:
        param_size: Model parameter size.
        attn_implementation: Attention implementation to use.
        dtype: Data type for model parameters.
        additional_model_config_kwargs: Additional model configuration parameters.
        checkpoint: Checkpoint to use for model initialization.
        use_device: Device to load the model onto.
    Returns:
        Initialized QiChatForCausalLM model.
    """
    # 根据 param_size 设置模型配置参数
    if param_size not in MODEL_PRESETS.keys(): raise ValueError(f'Unsupported model param_size: {param_size}')

    model_config_kwargs = MODEL_PRESETS[param_size]
    model_config_kwargs['attn_implementation'] = attn_implementation
    model_config_kwargs['dtype'] = dtype
    if additional_model_config_kwargs: model_config_kwargs.update(additional_model_config_kwargs)

    model_config = QiChatConfig(**model_config_kwargs)
    model = QiChatForCausalLM(config=model_config)
    model.to(dtype=model_config.dtype)

    if checkpoint is not None: model.load_state_dict(checkpoint)

    if use_device is not None:
        if isinstance(use_device, str):
            use_device = get_device(device=use_device)
        model.to(use_device)

    return model


def load_model_checkpoint(map_location: str | torch.device | None = None) -> dict:
    """Load the latest model checkpoint for inference."""
    checkpoint_path = data_path('checkpoint')
    checkpoint_list = [
        checkpoint for checkpoint in os.listdir(checkpoint_path)
        if checkpoint.startswith('CheckPoint_') and checkpoint.endswith('.pth')
    ]
    if not checkpoint_list: raise FileNotFoundError("No checkpoint found for inference.")

    # 自定义排序函数
    def extract_sort_key(filename):
        import re
        match = re.search(r'_(\d+\.\d+)_(\d+)_', filename)
        if match:
            epoch_batch_num = float(match.group(1))  # 7.800 -> 7.800
            total_batch = int(match.group(2))  # 30260 -> 30260
            return total_batch
        return 0

    checkpoint_list.sort(key=extract_sort_key)
    print(f'Loading checkpoint: {checkpoint_list[-1]}')
    checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint_list[-1]), map_location=map_location)
    return checkpoint


def generate_random_inputs(
    batch_size: int,
    seq_length: int,
    vocab_size: int | None = None,
    d_model: int | None = None,
    pad_token_id: int | None = None,
    pad_side: Literal['right', 'left'] = 'right',
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """
    生成随机的输入，兼容模型或模块。
    当提供 vocab_size 时，生成随机 input_ids；当提供 d_model 时，生成随机 hidden_states。
    如果不提供 pad_token_id，则 attention_mask 全为 True，否则随机生成每条样本的有效长度。

    Args:
        batch_size(int): Batch size
        seq_length(int): Sequence length
        vocab_size(int, optional): Vocabulary size for generating input_ids
        d_model(int, optional): Model dimension for generating hidden_states
        pad_token_id(int, optional): Padding token ID to use for masked positions (only relevant if vocab_size is provided)
        pad_side(Literal['right', 'left'], optional): Padding side for input_ids (only relevant if vocab_size is provided)
        dtype(torch.dtype): Data type of the generated tensors
        device(torch.device, optional): Device to place the generated tensors on
    Returns:
        dict[str, torch.Tensor]: A dictionary containing 'input_ids' and 'attention_mask' or 'hidden_states' and 'attention_mask'
    """
    if vocab_size is None and d_model is None:
        raise ValueError("Either vocab_size or d_model must be provided.")
    if vocab_size is not None and d_model is not None:
        raise ValueError("Only one of vocab_size or d_model should be provided.")

    if pad_token_id is not None:
        valid_len = torch.randint(1, seq_length + 1, (batch_size,), device=device)
        pos = torch.arange(seq_length, device=device).unsqueeze(0)  # [1, S]
        if pad_side == 'right': attn_mask = pos < valid_len.unsqueeze(1)  # [B, S] bool
        elif pad_side == 'left': attn_mask = pos >= (seq_length - valid_len).unsqueeze(1)  # [B, S] bool
        else: raise ValueError(f'Unsupported pad_side: {pad_side}')
    else: attn_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=device)

    return_dict = {
        'attention_mask': attn_mask,
    }
    if vocab_size is not None:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        labels = input_ids.clone()
        if not attn_mask.all():
            input_ids[~attn_mask] = pad_token_id
            labels[~attn_mask] = -100  # 将 masked 位置的 labels 置为 -100，确保 CrossEntropyLoss 在计算 loss 时忽略这些位置
        return_dict.update({
            'input_ids': input_ids,
            'labels': labels,
        })
    if d_model is not None:
        hidden_states = torch.randn((batch_size, seq_length, d_model), dtype=dtype, device=device)
        if not attn_mask.all(): hidden_states *= attn_mask.unsqueeze(-1)  # 将 masked 位置的 hidden_states 置零
        return_dict['hidden_states'] = hidden_states

    return return_dict
