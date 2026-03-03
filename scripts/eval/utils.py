import os
from typing import Literal

import torch
from lm_eval.models.huggingface import HFLM
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from src import root_path
from src.module import MODEL_PRESETS, QiChatConfig, QiChatForCausalLM
from src.tokenizer import get_tokenizer
from src.utils.path import data_path


def init_tokenizer() -> PreTrainedTokenizerFast:
    """ Initialize the tokenizer """
    tokenizer = get_tokenizer()
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single="$A <|eos|>",
        pair="<|bos|> $A <|eos|> $B <|eos|>",
        special_tokens=[
            (tokenizer.bos_token, tokenizer.bos_token_id),
            (tokenizer.eos_token, tokenizer.eos_token_id),
        ],
    )
    tokenizer.chat_template = """{% for message in messages -%}
        <|{{ message.role }}|>{{ message.content }}<|eot|>
        {%- if not loop.last -%}{{ '\n' }}{% endif %}
    {% endfor %}
    {% if add_generation_prompt -%}
        {{ '\n' }}<|assistant|>
        {%- if enable_thinking %}{{ '\n' }}<|begin_of_think|>{% endif %}
    {% endif %}"""

    return tokenizer


def init_eval_model(
    param_size: Literal['Tiny', '73M', '0.3B', '0.6B', '1.3B', '3.2B'] = 'Tiny',
    attn_implementation: Literal['eager', 'memory_efficient', 'sdpa', 'flash_attention_2', 'flash_attention_3'] = 'flash_attention_2',
    dtype: Literal['float16', 'bfloat16', 'float32'] | torch.dtype = 'bfloat16',
    additional_model_config_kwargs: dict | None = None,
    checkpoint_file: str | None = None,
    use_device: Literal['cuda', 'mps', 'cpu'] | torch.device | None = None,
) -> HFLM:
    """
    Initialize the model for evaluation
    Args:
        param_size: Model parameter size.
        attn_implementation: Attention implementation to use.
        dtype: Data type for model parameters.
        additional_model_config_kwargs: Additional model configuration parameters.
        checkpoint_file: Checkpoint file to use for model initialization.
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

    checkpoint_path = data_path('checkpoint')
    checkpoint_list = [
        checkpoint for checkpoint in os.listdir(checkpoint_path)
        if checkpoint.startswith('CheckPoint_') and checkpoint.endswith('.pth')
    ]
    if not checkpoint_list:
        raise FileNotFoundError(f'No checkpoint files found in {checkpoint_path.relative_to(root_path())}. Please run the pretraining script to generate checkpoints before evaluation.')

    if checkpoint_file is not None:
        if checkpoint_file not in checkpoint_list:
            raise FileNotFoundError(f'Checkpoint file {checkpoint_file} not found in {checkpoint_path.relative_to(root_path())}. Available checkpoints: {checkpoint_list}')
        checkpoint_path = data_path('checkpoint', checkpoint_file)
    else:
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
        checkpoint_path = data_path('checkpoint', checkpoint_list[-1])

    checkpoint = torch.load(checkpoint_path, map_location=use_device)['state']['weights']
    model.load_state_dict(checkpoint)

    if use_device is not None:
        if isinstance(use_device, str):
            use_device = torch.device(use_device)
        model.to(use_device)

    return HFLM(
        pretrained=model.eval(),
        backend='causal',
        max_length=model_config.max_position_embeddings,
        device=use_device,
        dtype=model_config.dtype,
        tokenizer=init_tokenizer(),
        use_fast_tokenizer=True,
    )
