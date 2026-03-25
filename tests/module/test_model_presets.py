import pytest

from src.module import MODEL_PRESETS, QiChatConfig


EXPECTED_PRESET_NAMES = {'Tiny', '73M', '0.3B', '0.6B', '1.3B', '3.2B'}


def test_model_presets_exposes_expected_sizes() -> None:
    assert set(MODEL_PRESETS.keys()) == EXPECTED_PRESET_NAMES


@pytest.mark.parametrize('preset_name', sorted(EXPECTED_PRESET_NAMES))
def test_model_presets_can_build_valid_configs(preset_name: str) -> None:
    preset = MODEL_PRESETS[preset_name]
    config = QiChatConfig(**preset, attn_implementation='eager', dtype='float32')

    assert config.d_model % 2 == 0
    assert config.d_model % config.n_heads == 0
    assert config.d_kv == config.d_model // config.n_heads
    assert 0 < config.n_kv_heads <= config.n_heads
    assert config.n_heads % config.n_kv_heads == 0
    assert config.max_position_embeddings == preset['max_position_embeddings']


def test_model_presets_top_level_mapping_is_immutable() -> None:
    with pytest.raises(TypeError):
        MODEL_PRESETS['Custom'] = MODEL_PRESETS['Tiny']

    with pytest.raises(TypeError):
        del MODEL_PRESETS['Tiny']
