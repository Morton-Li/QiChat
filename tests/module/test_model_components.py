import pytest
import torch
from transformers import DynamicCache
from transformers.masking_utils import create_causal_mask

from src.module.configuration import QiChatConfig
from src.module.model import (
    GLUFeedForward,
    QiChatAttention,
    QiChatBlock,
    QiChatRotaryPositionEmbedding,
    QiChatWordEmbeddings,
    apply_rotary_pos_emb,
)


def _build_cos_sin(config: QiChatConfig, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    rope = QiChatRotaryPositionEmbedding(
        dim=config.d_kv,
        base=config.rope_theta,
        max_position_embeddings=config.max_position_embeddings,
        dtype=config.dtype,
    )
    cos_buf, sin_buf = rope(position_ids)
    return position_ids, cos_buf, sin_buf


def test_word_embeddings_zero_masked_tokens_and_truncate_longer_mask() -> None:
    module = QiChatWordEmbeddings(vocab_size=8, d_model=4, pad_token_id=0, dropout_prob=0.0)
    with torch.no_grad():
        module.word_embeddings.weight.copy_(torch.arange(32, dtype=torch.float32).view(8, 4))

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    long_mask = torch.tensor([[True, True, False, True, False]], dtype=torch.bool)

    outputs = module(input_ids=input_ids, mask=long_mask)

    expected = module.word_embeddings(input_ids)
    expected[:, 0] = 0.0
    expected[:, 2] = 0.0
    torch.testing.assert_close(outputs, expected)


def test_rotary_position_embedding_cached_and_dynamic_paths_match() -> None:
    position_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    dynamic_rope = QiChatRotaryPositionEmbedding(dim=16, base=10000, max_position_embeddings=None, dtype=torch.float32)
    cached_rope = QiChatRotaryPositionEmbedding(dim=16, base=10000, max_position_embeddings=8, dtype=torch.float32)

    dynamic_cos, dynamic_sin = dynamic_rope(position_ids)
    cached_cos, cached_sin = cached_rope(position_ids)

    assert dynamic_cos.dtype == torch.float32
    assert dynamic_sin.dtype == torch.float32
    torch.testing.assert_close(dynamic_cos, cached_cos, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(dynamic_sin, cached_sin, atol=1e-6, rtol=1e-6)


def test_rotary_position_embedding_validates_position_ids_rank() -> None:
    rope = QiChatRotaryPositionEmbedding(dim=16, dtype=torch.float32)

    with pytest.raises(ValueError, match='position_ids must be of shape'):
        rope(torch.arange(4, dtype=torch.long))


def test_apply_rotary_pos_emb_identity_and_shape_validation() -> None:
    x = torch.randn(2, 3, 4, 8)
    cos = torch.ones(2, 4, 4)
    sin = torch.zeros(2, 4, 4)

    rotated = apply_rotary_pos_emb(x=x, cos=cos, sin=sin)
    torch.testing.assert_close(rotated, x)

    with pytest.raises(ValueError, match='cos and sin must have shape'):
        apply_rotary_pos_emb(x=x, cos=torch.ones(2, 4, 5), sin=torch.zeros(2, 4, 5))


def test_attention_forward_returns_expected_shapes_and_weights(tiny_eager_config: QiChatConfig) -> None:
    attention = QiChatAttention(config=tiny_eager_config, layer_idx=0).eval()
    hidden_states = torch.randn(2, 5, tiny_eager_config.d_model, dtype=tiny_eager_config.dtype)
    attention_mask = create_causal_mask(
        config=tiny_eager_config,
        inputs_embeds=hidden_states,
        attention_mask=torch.ones(2, 5, dtype=torch.bool),
        cache_position=torch.arange(5, dtype=torch.long),
        past_key_values=None,
    )
    _, cos_buf, sin_buf = _build_cos_sin(tiny_eager_config, batch_size=2, seq_len=5)

    attn_output, attn_weights = attention(
        hidden_states=hidden_states,
        mask=attention_mask,
        cos_buf=cos_buf,
        sin_buf=sin_buf,
        output_attentions=True,
    )

    assert attn_output.shape == (2, 5, tiny_eager_config.d_model)
    assert attn_weights is not None
    assert attn_weights.shape == (2, tiny_eager_config.n_heads, 5, 5)


def test_attention_matches_between_eager_and_sdpa(tiny_eager_config: QiChatConfig, tiny_sdpa_config: QiChatConfig) -> None:
    eager_attention = QiChatAttention(config=tiny_eager_config, layer_idx=0).eval()
    sdpa_attention = QiChatAttention(config=tiny_sdpa_config, layer_idx=0).eval()
    sdpa_attention.load_state_dict(eager_attention.state_dict())

    hidden_states = torch.randn(2, 5, tiny_eager_config.d_model, dtype=tiny_eager_config.dtype)
    attention_mask = torch.ones(2, 5, dtype=torch.bool)
    eager_mask = create_causal_mask(
        config=tiny_eager_config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        cache_position=torch.arange(5, dtype=torch.long),
        past_key_values=None,
    )
    sdpa_mask = create_causal_mask(
        config=tiny_sdpa_config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        cache_position=torch.arange(5, dtype=torch.long),
        past_key_values=None,
    )
    _, cos_buf, sin_buf = _build_cos_sin(tiny_eager_config, batch_size=2, seq_len=5)

    eager_output, _ = eager_attention(hidden_states=hidden_states, mask=eager_mask, cos_buf=cos_buf, sin_buf=sin_buf)
    sdpa_output, _ = sdpa_attention(hidden_states=hidden_states, mask=sdpa_mask, cos_buf=cos_buf, sin_buf=sin_buf)

    torch.testing.assert_close(eager_output, sdpa_output, atol=1e-5, rtol=1e-5)


def test_attention_validates_dtype_and_cache_arguments(tiny_eager_config: QiChatConfig) -> None:
    attention = QiChatAttention(config=tiny_eager_config, layer_idx=0).eval()
    hidden_states = torch.randn(2, 4, tiny_eager_config.d_model, dtype=tiny_eager_config.dtype)
    position_ids, cos_buf, sin_buf = _build_cos_sin(tiny_eager_config, batch_size=2, seq_len=4)

    with pytest.raises(ValueError, match='same dtype'):
        attention(
            hidden_states=hidden_states,
            mask=None,
            cos_buf=cos_buf.to(dtype=torch.float64),
            sin_buf=sin_buf.to(dtype=torch.float64),
        )

    with pytest.raises(ValueError, match='past_key_values must be a Cache object'):
        attention(
            hidden_states=hidden_states,
            mask=None,
            cos_buf=cos_buf,
            sin_buf=sin_buf,
            use_cache=True,
            position_ids=position_ids,
            past_key_values='invalid',
            cache_position=torch.arange(4, dtype=torch.long),
        )

    with pytest.raises(ValueError, match='cache_position must be a LongTensor'):
        attention(
            hidden_states=hidden_states,
            mask=None,
            cos_buf=cos_buf,
            sin_buf=sin_buf,
            use_cache=True,
            position_ids=position_ids,
            past_key_values=DynamicCache(config=tiny_eager_config),
            cache_position=torch.arange(4, dtype=torch.int32),
        )


def test_attention_rejects_use_cache_while_training(tiny_eager_config: QiChatConfig) -> None:
    attention = QiChatAttention(config=tiny_eager_config, layer_idx=0).train()
    _, cos_buf, sin_buf = _build_cos_sin(tiny_eager_config, batch_size=1, seq_len=3)

    with pytest.raises(RuntimeError, match='use_cache=True is not supported during training'):
        attention(
            hidden_states=torch.randn(1, 3, tiny_eager_config.d_model, dtype=tiny_eager_config.dtype),
            mask=None,
            cos_buf=cos_buf,
            sin_buf=sin_buf,
            use_cache=True,
        )


def test_glu_feed_forward_shape_and_activation_validation() -> None:
    module = GLUFeedForward(d_model=16, d_ff=32, activation='silu')
    outputs = module(torch.randn(2, 5, 16))
    assert outputs.shape == (2, 5, 16)

    with pytest.raises(ValueError, match='activation unsupported is not supported'):
        GLUFeedForward(d_model=16, d_ff=32, activation='unsupported')


def test_block_forward_returns_hidden_states_and_attentions(tiny_eager_config: QiChatConfig) -> None:
    block = QiChatBlock(config=tiny_eager_config, layer_idx=0).eval()
    hidden_states = torch.randn(2, 5, tiny_eager_config.d_model, dtype=tiny_eager_config.dtype)
    mask = create_causal_mask(
        config=tiny_eager_config,
        inputs_embeds=hidden_states,
        attention_mask=torch.ones(2, 5, dtype=torch.bool),
        cache_position=torch.arange(5, dtype=torch.long),
        past_key_values=None,
    )
    position_ids, cos_buf, sin_buf = _build_cos_sin(tiny_eager_config, batch_size=2, seq_len=5)

    outputs, attn_weights = block(
        hidden_states=hidden_states,
        mask=mask,
        cos_buf=cos_buf,
        sin_buf=sin_buf,
        use_cache=False,
        position_ids=position_ids,
        output_attentions=True,
    )

    assert outputs.shape == (2, 5, tiny_eager_config.d_model)
    assert attn_weights is not None
    assert attn_weights.shape == (2, tiny_eager_config.n_heads, 5, 5)
