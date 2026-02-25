from types import MappingProxyType
from typing import Literal, TypedDict


ModelSize = Literal["Tiny", "73M", "0.3B", "0.6B", "1.3B", "3.2B"]


class ModelPreset(TypedDict):
    d_model: int
    num_hidden_layers: int
    n_heads: int
    d_kv: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int


_PRESETS: dict[ModelSize, ModelPreset] = {
    'Tiny': {  # 2.17 M, 用于测试和调试
        'd_model': 64, 'num_hidden_layers': 2, 'n_heads': 2, 'd_kv': 32, 'n_kv_heads': 1, 'd_ff': 256, 'vocab_size': 32000,
    },
    '73M': {  # 73.27 M, 标准多头注意力（MHA）, 用于可行性验证
        'd_model': 512, 'num_hidden_layers': 16, 'n_heads': 8, 'd_kv': 64, 'n_kv_heads': 8, 'd_ff': 2048, 'vocab_size': 12000,
    },
    '0.3B': {  # 347.38 M, 使用 SwiGLU 但依旧执行 4*d_model 以获得更高性能, GQA 2:1, 目标上下文上限为 2048
        'd_model': 1024, 'num_hidden_layers': 20, 'n_heads': 16, 'd_kv': 64, 'n_kv_heads': 8, 'd_ff': 4096, 'vocab_size': 32000,
    },
    '0.6B': {  # 672.08 M, 使用 SwiGLU, d_ff 调整为近似 2.67*d_model, GQA 2:1, 目标上下文上限为 4096
        'd_model': 1536, 'num_hidden_layers': 24, 'n_heads': 24, 'd_kv': 64, 'n_kv_heads': 12, 'd_ff': 4096, 'vocab_size': 32000,
    },
    '1.3B': {  # 1328.14 M, GQA 4:1, SwiGLU, 目标上下文上限为 5120
        'd_model': 2048, 'num_hidden_layers': 28, 'n_heads': 32, 'd_kv': 64, 'n_kv_heads': 8, 'd_ff': 5632, 'vocab_size': 32000,
    },
    '3.2B': {  # 3269.4 M, GQA 4:1, SwiGLU, 目标上下文上限为 8192
        'd_model': 3072, 'num_hidden_layers': 32, 'n_heads': 48, 'd_kv': 64, 'n_kv_heads': 12, 'd_ff': 8192, 'vocab_size': 32000,
    },
}

MODEL_PRESETS = MappingProxyType(_PRESETS)
