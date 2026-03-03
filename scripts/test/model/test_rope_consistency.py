import random
from typing import Literal

import torch

from src.module.model import QiChatRotaryPositionEmbedding


def build_random_position_ids(
    batch_size: int,
    seq_len: int,
    padding_side: Literal['left', 'right'] | None = None,
    padding_value: int = 1,  # HF 的实现为 1
) -> torch.LongTensor:
    """
    构建随机位置 ID 张量，支持可变长度输入和左右填充。
    Args:
        batch_size: 批次大小。
        seq_len: 序列长度。
        padding_side: 填充方向，'left' 或 'right'，如果为 None 则不进行填充。
        padding_value: 填充值，默认为 1（HF 的实现）。
    Returns:
        position_ids: 形状为 (batch_size, seq_len) 的位置 ID 张量。
    """
    position_ids = torch.empty(batch_size, seq_len, dtype=torch.long)
    for i in range(batch_size):
        valid_len = random.randint(1, seq_len) if padding_side is not None else seq_len
        position_ids[i, :valid_len] = torch.arange(valid_len)
        if padding_side == 'right': position_ids[i, valid_len:] = padding_value
        elif padding_side == 'left': position_ids[i, :valid_len] = padding_value
    return position_ids


def test(
    batch_size: int = 16,
    seq_len: int = 4096,
    d_kv: int = 128,
    rope_base: int = 10000,
    use_dtype: torch.dtype = torch.bfloat16,
    use_device: torch.device = torch.device("mps"),
):
    with torch.no_grad():
        # 动态计算版 RoPE
        rope_dynamic = QiChatRotaryPositionEmbedding(dim=d_kv, base=rope_base, dtype=use_dtype).to(use_device).eval()
        # 缓存版 RoPE
        rope_cache = QiChatRotaryPositionEmbedding(dim=d_kv, base=rope_base, dtype=use_dtype, max_position_embeddings=5120).to(use_device).eval()

        position_ids = build_random_position_ids(batch_size, seq_len, padding_side='right').to(use_device)

        cos_dynamic, sin_dynamic = rope_dynamic(position_ids)
        cos_cache, sin_cache = rope_cache(position_ids)

        dcos = (cos_dynamic.float() - cos_cache.float()).abs()
        dsin = (sin_dynamic.float() - sin_cache.float()).abs()

    print(f"Max abs diff in cos: {dcos.max().item():.6f}, mean abs diff in cos: {dcos.mean().item():.6f}")
    print(f"Max abs diff in sin: {dsin.max().item():.6f}, mean abs diff in sin: {dsin.mean().item():.6f}")


if __name__ == "__main__":
    test()
