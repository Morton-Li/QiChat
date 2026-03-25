import pytest
import torch

from src.module import MODEL_PRESETS, QiChatConfig


def test_configuration_initializes_expected_defaults() -> None:
    config = QiChatConfig(**MODEL_PRESETS['Tiny'], attn_implementation='eager', dtype='float32')

    assert config.model_type == 'QiChat'
    assert config.dtype == torch.float32
    assert config.return_dict is True
    assert config.tokenizer_class == 'QiTianTokenizerFast'
    assert config.keys_to_ignore_at_inference == ['past_key_values']
    assert config.base_model_pp_plan['word_emb'] == (['input_ids', 'mask'], ['hidden_states'])
    assert len(config.layer_types) == config.num_hidden_layers
    assert all(layer_type == 'full_attention' for layer_type in config.layer_types)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'d_model': 63, 'n_heads': 3, 'd_kv': 21}, 'd_model must be an even number'),
        ({'d_model': 64, 'n_heads': 3, 'd_kv': 21}, 'd_model must be divisible by num_heads'),
        ({'d_model': 64, 'n_heads': 4, 'd_kv': 8}, 'd_kv must equal d_model // n_heads'),
        ({'d_model': 64, 'n_heads': 4, 'd_kv': 16, 'n_kv_heads': 5}, 'n_kv_heads must be less than or equal to num_heads'),
        ({'d_model': 64, 'n_heads': 4, 'd_kv': 16, 'n_kv_heads': 0}, 'n_kv_heads must be greater than 0'),
        ({'d_model': 72, 'n_heads': 6, 'd_kv': 12, 'n_kv_heads': 4}, 'num_heads must be divisible by n_kv_heads'),
    ],
)
def test_configuration_validates_core_shape_constraints(kwargs: dict[str, int], message: str) -> None:
    base_kwargs = {
        'vocab_size': 128,
        'd_model': 64,
        'num_hidden_layers': 2,
        'n_heads': 4,
        'd_kv': 16,
        'n_kv_heads': 2,
        'd_ff': 128,
        'attn_implementation': 'eager',
        'dtype': 'float32',
    }
    base_kwargs.update(kwargs)

    with pytest.raises(AssertionError, match=message):
        QiChatConfig(**base_kwargs)


def test_configuration_requires_explicit_attn_implementation_when_loading_from_pretrained(tmp_path) -> None:
    config = QiChatConfig(**MODEL_PRESETS['Tiny'], attn_implementation='sdpa', dtype='float32')
    save_dir = tmp_path / 'config'

    config.save_pretrained(save_dir)
    reloaded = QiChatConfig.from_pretrained(save_dir, attn_implementation='sdpa')

    assert reloaded.dtype == torch.float32
    assert reloaded._attn_implementation == 'sdpa'
    assert reloaded.to_dict()['vocab_size'] == config.vocab_size
    assert reloaded.to_dict()['d_model'] == config.d_model
