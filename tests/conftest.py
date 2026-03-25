import random

import numpy
import pytest
import torch

from src.module import MODEL_PRESETS, QiChatConfig
from src.module.model import QiChatForCausalLM, QiChatModel

SEED = 36


@pytest.fixture(autouse=True)
def reset_random_seed() -> None:
    random.seed(SEED)
    numpy.random.seed(SEED)
    torch.manual_seed(SEED)


@pytest.fixture
def tiny_eager_config() -> QiChatConfig:
    return QiChatConfig(
        **MODEL_PRESETS['Tiny'],
        attn_implementation='eager',
        dtype='float32',
    )


@pytest.fixture
def tiny_sdpa_config() -> QiChatConfig:
    return QiChatConfig(
        **MODEL_PRESETS['Tiny'],
        attn_implementation='sdpa',
        dtype='float32',
    )


@pytest.fixture
def tiny_model(tiny_eager_config: QiChatConfig) -> QiChatModel:
    return QiChatModel(tiny_eager_config).eval()


@pytest.fixture
def tiny_lm_model(tiny_eager_config: QiChatConfig) -> QiChatForCausalLM:
    return QiChatForCausalLM(tiny_eager_config).eval()


@pytest.fixture
def sample_batch(tiny_eager_config: QiChatConfig) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [4, 5, 6, 7, tiny_eager_config.pad_token_id, tiny_eager_config.pad_token_id],
            [8, 9, 10, 11, 12, 13],
        ],
        dtype=torch.long,
    )
    attention_mask = input_ids != tiny_eager_config.pad_token_id
    labels = input_ids.clone()
    labels[~attention_mask] = -100
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


@pytest.fixture
def non_padded_batch() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    labels = input_ids.clone()
    labels[~attention_mask] = -100
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


@pytest.fixture
def left_right_padded_inputs(tiny_eager_config: QiChatConfig) -> dict[str, torch.Tensor]:
    sequences = [
        torch.tensor([7, 8, 9], dtype=torch.long),
        torch.tensor([10, 11], dtype=torch.long),
        torch.tensor([12, 13, 14, 15], dtype=torch.long),
    ]
    pad = tiny_eager_config.pad_token_id
    seq_len = 6

    left_rows = []
    right_rows = []
    for sequence in sequences:
        pad_len = seq_len - sequence.numel()
        pad_tokens = torch.full((pad_len,), pad, dtype=torch.long)
        left_rows.append(torch.cat([pad_tokens, sequence], dim=0))
        right_rows.append(torch.cat([sequence, pad_tokens], dim=0))

    left_input_ids = torch.stack(left_rows)
    right_input_ids = torch.stack(right_rows)
    left_attention_mask = left_input_ids != pad
    right_attention_mask = right_input_ids != pad

    indices = torch.arange(seq_len, dtype=torch.long)
    left_last_valid_idx = (left_attention_mask.long() * indices).amax(dim=-1)
    right_last_valid_idx = (right_attention_mask.long() * indices).amax(dim=-1)

    return {
        'left_input_ids': left_input_ids,
        'left_attention_mask': left_attention_mask,
        'right_input_ids': right_input_ids,
        'right_attention_mask': right_attention_mask,
        'left_last_valid_idx': left_last_valid_idx,
        'right_last_valid_idx': right_last_valid_idx,
    }

