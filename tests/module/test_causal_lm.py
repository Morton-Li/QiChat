import pytest
import torch
from torch import nn
from transformers import DynamicCache

from src.module.model import QiChatCausalLMHead, QiChatForCausalLM


def test_lm_head_projects_to_vocab_and_validates_hidden_size() -> None:
	head = QiChatCausalLMHead(d_model=16, vocab_size=32)
	logits = head(torch.randn(2, 5, 16))
	assert logits.shape == (2, 5, 32)

	with pytest.raises(AssertionError, match='Expected hidden_states last dimension to be 16'):
		head(torch.randn(2, 5, 17))


def test_causal_lm_forward_computes_expected_shifted_loss(
	tiny_lm_model: QiChatForCausalLM,
	sample_batch: dict[str, torch.Tensor],
) -> None:
	outputs = tiny_lm_model(
		input_ids=sample_batch['input_ids'],
		attention_mask=sample_batch['attention_mask'],
		labels=sample_batch['labels'],
		output_hidden_states=True,
		output_attentions=True,
	)

	shift_logits = outputs.logits[:, :-1, :].reshape(-1, tiny_lm_model.config.vocab_size)
	shift_labels = sample_batch['labels'][:, 1:].reshape(-1)
	expected_loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

	assert outputs.loss is not None
	assert outputs.hidden_states is not None
	assert outputs.attentions is not None
	assert len(outputs.hidden_states) == tiny_lm_model.config.num_hidden_layers + 1
	assert len(outputs.attentions) == tiny_lm_model.config.num_hidden_layers
	torch.testing.assert_close(outputs.loss, expected_loss, atol=1e-6, rtol=1e-6)


def test_causal_lm_validates_arguments(
	tiny_lm_model: QiChatForCausalLM,
	non_padded_batch: dict[str, torch.Tensor],
) -> None:
	input_ids = non_padded_batch['input_ids']
	attention_mask = non_padded_batch['attention_mask']
	inputs_embeds = tiny_lm_model.get_input_embeddings()(input_ids)

	with pytest.raises(ValueError, match='both input_ids and inputs_embeds'):
		tiny_lm_model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

	with pytest.raises(ValueError, match='either input_ids or inputs_embeds'):
		tiny_lm_model(attention_mask=attention_mask)

	with pytest.raises(NotImplementedError, match='return_dict=False is not supported yet'):
		tiny_lm_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)


def test_causal_lm_ties_input_and_output_embeddings(tiny_lm_model: QiChatForCausalLM) -> None:
	assert tiny_lm_model.get_input_embeddings().weight.data_ptr() == tiny_lm_model.get_output_embeddings().weight.data_ptr()


def test_causal_lm_cache_matches_full_forward(
	tiny_lm_model: QiChatForCausalLM,
	non_padded_batch: dict[str, torch.Tensor],
) -> None:
	input_ids = non_padded_batch['input_ids'][:1]
	attention_mask = non_padded_batch['attention_mask'][:1]

	full_outputs = tiny_lm_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

	prefill = tiny_lm_model(
		input_ids=input_ids[:, :3],
		attention_mask=attention_mask[:, :3],
		use_cache=True,
		past_key_values=DynamicCache(config=tiny_lm_model.config),
	)
	step_1 = tiny_lm_model(
		input_ids=input_ids[:, 3:4],
		attention_mask=attention_mask[:, :4],
		use_cache=True,
		past_key_values=prefill.past_key_values,
	)
	step_2 = tiny_lm_model(
		input_ids=input_ids[:, 4:5],
		attention_mask=attention_mask,
		use_cache=True,
		past_key_values=step_1.past_key_values,
	)

	assert step_2.past_key_values is not None
	assert step_2.past_key_values.get_seq_length() == input_ids.size(1)
	torch.testing.assert_close(full_outputs.logits[:, 3:4], step_1.logits, atol=1e-6, rtol=1e-6)
	torch.testing.assert_close(full_outputs.logits[:, 4:5], step_2.logits, atol=1e-6, rtol=1e-6)


def test_causal_lm_padding_invariance_on_last_valid_token(
	tiny_lm_model: QiChatForCausalLM,
	left_right_padded_inputs: dict[str, torch.Tensor],
) -> None:
	batch_indices = torch.arange(left_right_padded_inputs['left_input_ids'].size(0), dtype=torch.long)

	left_outputs = tiny_lm_model(
		input_ids=left_right_padded_inputs['left_input_ids'],
		attention_mask=left_right_padded_inputs['left_attention_mask'],
	)
	right_outputs = tiny_lm_model(
		input_ids=left_right_padded_inputs['right_input_ids'],
		attention_mask=left_right_padded_inputs['right_attention_mask'],
	)

	left_logits = left_outputs.logits[batch_indices, left_right_padded_inputs['left_last_valid_idx']]
	right_logits = right_outputs.logits[batch_indices, left_right_padded_inputs['right_last_valid_idx']]
	torch.testing.assert_close(left_logits, right_logits, atol=1e-5, rtol=1e-5)


def test_causal_lm_prevents_future_leakage(
	tiny_lm_model: QiChatForCausalLM,
	non_padded_batch: dict[str, torch.Tensor],
) -> None:
	input_ids = non_padded_batch['input_ids'].clone()
	attention_mask = non_padded_batch['attention_mask']

	original_logits = tiny_lm_model(input_ids=input_ids, attention_mask=attention_mask).logits
	modified_input_ids = input_ids.clone()
	modified_input_ids[:, -1] = modified_input_ids[:, -1] + 5
	modified_logits = tiny_lm_model(input_ids=modified_input_ids, attention_mask=attention_mask).logits

	torch.testing.assert_close(original_logits[:, -2], modified_logits[:, -2], atol=1e-6, rtol=1e-6)


def test_causal_lm_training_rejects_use_cache(
	tiny_lm_model: QiChatForCausalLM,
	non_padded_batch: dict[str, torch.Tensor],
) -> None:
	tiny_lm_model.train()

	with pytest.raises(RuntimeError, match='use_cache=True cannot be used during training'):
		tiny_lm_model(**non_padded_batch, use_cache=True)
