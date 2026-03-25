import torch

from src.module import QiChatConfig
from src.module.model import QiChatForCausalLM, QiChatModel


def test_model_state_dict_round_trip_preserves_outputs(
    tiny_eager_config: QiChatConfig,
    sample_batch: dict[str, torch.Tensor],
) -> None:
    model = QiChatModel(tiny_eager_config).eval()
    cloned_model = QiChatModel(tiny_eager_config).eval()
    cloned_model.load_state_dict(model.state_dict())

    original = model(
        input_ids=sample_batch['input_ids'],
        attention_mask=sample_batch['attention_mask'],
    )
    cloned = cloned_model(
        input_ids=sample_batch['input_ids'],
        attention_mask=sample_batch['attention_mask'],
    )

    torch.testing.assert_close(original.last_hidden_state, cloned.last_hidden_state, atol=1e-6, rtol=1e-6)


def test_causal_lm_save_pretrained_round_trip_preserves_outputs_and_tying(
    tiny_lm_model: QiChatForCausalLM,
    tmp_path,
) -> None:
    save_dir = tmp_path / 'qichat-lm'
    tiny_lm_model.save_pretrained(save_dir)
    reloaded = QiChatForCausalLM.from_pretrained(
        save_dir,
        attn_implementation=tiny_lm_model.config._attn_implementation,
    ).eval()

    baseline_state_dict = tiny_lm_model.state_dict()
    reloaded_state_dict = reloaded.state_dict()
    assert baseline_state_dict.keys() == reloaded_state_dict.keys()
    for key in baseline_state_dict:
        torch.testing.assert_close(baseline_state_dict[key], reloaded_state_dict[key], atol=1e-6, rtol=1e-6)

    assert reloaded.config.dtype == torch.float32
    assert reloaded.config._attn_implementation == tiny_lm_model.config._attn_implementation
    assert reloaded.get_input_embeddings().weight.data_ptr() == reloaded.get_output_embeddings().weight.data_ptr()
