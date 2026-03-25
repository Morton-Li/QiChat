import pytest
import torch
from transformers import DynamicCache

from src.module.configuration import QiChatConfig
from src.module.model import QiChatModel


def test_model_dummy_inputs_have_expected_shape_and_types(tiny_model: QiChatModel) -> None:
    dummy_inputs = tiny_model.dummy_inputs

    assert set(dummy_inputs.keys()) == {'input_ids', 'attention_mask'}
    assert dummy_inputs['input_ids'].dtype == torch.long
    assert dummy_inputs['attention_mask'].dtype == torch.bool
    assert dummy_inputs['input_ids'].shape == dummy_inputs['attention_mask'].shape == (3, 12)


def test_model_forward_returns_hidden_states_and_attentions(
    tiny_model: QiChatModel,
    non_padded_batch: dict[str, torch.Tensor],
) -> None:
    non_padded_batch.pop('labels')
    outputs = tiny_model(
        **non_padded_batch,
        output_hidden_states=True,
        output_attentions=True,
    )

    assert outputs.last_hidden_state.shape == (2, 5, tiny_model.config.d_model)
    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == tiny_model.config.num_hidden_layers + 1
    assert outputs.attentions is not None
    assert len(outputs.attentions) == tiny_model.config.num_hidden_layers
    assert outputs.attentions[0].shape == (2, tiny_model.config.n_heads, 5, 5)


def test_model_forward_returns_none_for_optional_outputs_when_disabled(
    tiny_model: QiChatModel,
    non_padded_batch: dict[str, torch.Tensor],
) -> None:
    non_padded_batch.pop('labels')
    outputs = tiny_model(
        **non_padded_batch,
        output_hidden_states=False,
        output_attentions=False,
    )

    assert outputs.hidden_states is None
    assert outputs.attentions is None


def test_model_forward_supports_inputs_embeds_equivalent_to_input_ids(
    tiny_model: QiChatModel,
    non_padded_batch: dict[str, torch.Tensor],
) -> None:
    input_ids = non_padded_batch['input_ids']
    attention_mask = non_padded_batch['attention_mask']
    inputs_embeds = tiny_model.get_input_embeddings()(input_ids)

    outputs_from_ids = tiny_model(input_ids=input_ids, attention_mask=attention_mask)
    outputs_from_embeds = tiny_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    torch.testing.assert_close(outputs_from_ids.last_hidden_state, outputs_from_embeds.last_hidden_state, atol=1e-6, rtol=1e-6)


def test_model_forward_casts_non_bool_attention_mask(
    tiny_model: QiChatModel,
    sample_batch: dict[str, torch.Tensor],
) -> None:
    bool_outputs = tiny_model(
        input_ids=sample_batch['input_ids'],
        attention_mask=sample_batch['attention_mask'],
    )
    int_outputs = tiny_model(
        input_ids=sample_batch['input_ids'],
        attention_mask=sample_batch['attention_mask'].long(),
    )

    torch.testing.assert_close(bool_outputs.last_hidden_state, int_outputs.last_hidden_state, atol=1e-6, rtol=1e-6)


def test_model_forward_validates_arguments(
    tiny_model: QiChatModel,
    non_padded_batch: dict[str, torch.Tensor],
) -> None:
    input_ids = non_padded_batch['input_ids']
    attention_mask = non_padded_batch['attention_mask']
    inputs_embeds = tiny_model.get_input_embeddings()(input_ids)

    with pytest.raises(ValueError, match='both input_ids and inputs_embeds'):
        tiny_model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    with pytest.raises(ValueError, match='either input_ids or inputs_embeds'):
        tiny_model(attention_mask=attention_mask)

    with pytest.raises(ValueError, match='does not match the model\'s d_model'):
        tiny_model(inputs_embeds=torch.randn(2, 5, tiny_model.config.d_model + 1), attention_mask=attention_mask)

    with pytest.raises(ValueError, match='does not match model dtype'):
        tiny_model(inputs_embeds=inputs_embeds.to(dtype=torch.float64), attention_mask=attention_mask)


def test_model_forward_supports_cache_and_matches_full_forward(
    tiny_model: QiChatModel,
    non_padded_batch: dict[str, torch.Tensor],
) -> None:
    input_ids = non_padded_batch['input_ids'][:1]
    attention_mask = non_padded_batch['attention_mask'][:1]

    full_outputs = tiny_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    prefill = tiny_model(
        input_ids=input_ids[:, :3],
        attention_mask=attention_mask[:, :3],
        use_cache=True,
        past_key_values=DynamicCache(config=tiny_model.config),
    )
    step_1 = tiny_model(
        input_ids=input_ids[:, 3:4],
        attention_mask=attention_mask[:, :4],
        use_cache=True,
        past_key_values=prefill.past_key_values,
    )
    step_2 = tiny_model(
        input_ids=input_ids[:, 4:5],
        attention_mask=attention_mask,
        use_cache=True,
        past_key_values=step_1.past_key_values,
    )

    assert step_2.past_key_values is not None
    assert step_2.past_key_values.get_seq_length() == input_ids.size(1)
    torch.testing.assert_close(full_outputs.last_hidden_state[:, 3:4], step_1.last_hidden_state, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(full_outputs.last_hidden_state[:, 4:5], step_2.last_hidden_state, atol=1e-6, rtol=1e-6)


def test_model_forward_matches_between_eager_and_sdpa(
    tiny_eager_config: QiChatConfig,
    tiny_sdpa_config: QiChatConfig,
    sample_batch: dict[str, torch.Tensor],
) -> None:
    eager_model = QiChatModel(tiny_eager_config).eval()
    sdpa_model = QiChatModel(tiny_sdpa_config).eval()
    sdpa_model.load_state_dict(eager_model.state_dict())

    eager_outputs = eager_model(
        input_ids=sample_batch['input_ids'],
        attention_mask=sample_batch['attention_mask'],
    )
    sdpa_outputs = sdpa_model(
        input_ids=sample_batch['input_ids'],
        attention_mask=sample_batch['attention_mask'],
    )

    torch.testing.assert_close(eager_outputs.last_hidden_state, sdpa_outputs.last_hidden_state, atol=1e-5, rtol=1e-5)


def test_model_training_rejects_use_cache(
    tiny_model: QiChatModel,
    non_padded_batch: dict[str, torch.Tensor],
) -> None:
    non_padded_batch.pop('labels')
    tiny_model.train()

    with pytest.raises(RuntimeError, match='use_cache=True cannot be used during training'):
        tiny_model(**non_padded_batch, use_cache=True)
