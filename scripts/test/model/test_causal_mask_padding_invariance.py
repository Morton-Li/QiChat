"""
模型因果注意力填充不变性测试

本脚本用于验证模型因果注意力机制是否不受输入序列中填充位置的影响。通过比较不同填充位置（左/右填充）的输出结果，确保模型能够正确地忽略填充部分，保持输出一致性。
"""
import random

import torch

from scripts.test.utils import set_seed, get_device, init_model, load_model_checkpoint


def test():
    set_seed(seed=36)
    use_device = get_device(device='auto')
    model = init_model(
        param_size='73M',
        attn_implementation='sdpa',
        dtype='bfloat16',
        use_device=use_device,
        checkpoint=load_model_checkpoint(map_location=use_device)['state']['weights'],
    ).eval()

    batch_size = 3
    pad_token_id = 3
    seq_length = 16
    source_inputs = []
    for _ in range(batch_size):
        seq_len = random.randint(1, seq_length)
        source_input_ids = torch.randint(20, 12000, (1, seq_len), device=use_device)
        pad_tensor = torch.ones((1, seq_length - seq_len), dtype=torch.long, device=use_device) * pad_token_id
        source_inputs.append((source_input_ids, pad_tensor))

    left_inputs = torch.cat([torch.cat([pad_i, ids_i], dim=1) for ids_i, pad_i in source_inputs], dim=0)
    left_mask = (left_inputs != pad_token_id)
    right_inputs = torch.cat([torch.cat([ids_i, pad_i], dim=1) for ids_i, pad_i in source_inputs], dim=0)
    right_mask = (right_inputs != pad_token_id)

    idx = torch.arange(seq_length, device=use_device)
    left_last_idx = (left_mask.long() * idx).amax(dim=-1)  # [Batch_size]
    right_last_idx = (right_mask.long() * idx).amax(dim=-1)  # [Batch_size]
    batch_indices = torch.arange(batch_size, device=use_device)

    with torch.no_grad():
        left_logits = model(
            input_ids=left_inputs,
            attention_mask=left_mask,
        ).logits[batch_indices, left_last_idx, :].detach()  # [batch_size, vocab_size]
        left_outputs = torch.argmax(left_logits, dim=-1)  # [batch_size]
        left_outputs_trimmed: list[int] = left_outputs.to("cpu").tolist()

        right_logits = model(
            input_ids=right_inputs,
            attention_mask=right_mask,
        ).logits[batch_indices, right_last_idx, :].detach()  # [batch_size, vocab_size]
        right_outputs = torch.argmax(right_logits, dim=-1)  # [batch_size]
        right_outputs_trimmed: list[int] = right_outputs.to("cpu").tolist()

    assert left_outputs_trimmed == right_outputs_trimmed, "Causal mask padding invariance test failed! The model's output differs between left-padded and right-padded inputs, indicating that the causal mask may not be properly ignoring the padding tokens."


if __name__ == '__main__':
    test()
