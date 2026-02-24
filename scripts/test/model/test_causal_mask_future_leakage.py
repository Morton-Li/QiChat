"""
模型因果注意力未来泄漏测试脚本

本脚本用于验证模型因果注意力机制是否有效确保模型不会错误的“看见未来”，原理是监测当前位置 token logits，在改动未来 token 时不应影响监测位置 logits
"""
import random

import torch

from scripts.test.utils import generate_random_inputs, set_seed, get_device, init_model


def test():
    set_seed(seed=36)
    use_device = get_device(device='auto')
    model = init_model(
        param_size='Tiny',
        attn_implementation='sdpa',
        dtype='bfloat16',
        use_device=use_device,
    ).eval()
    random_inputs = generate_random_inputs(
        batch_size=3,
        seq_length=16,
        vocab_size=12000,
        pad_token_id=None,
        dtype=torch.float32,
        device=use_device,
    )

    # 原始输入
    original_input_ids = random_inputs['input_ids']
    original_attention_mask = random_inputs['attention_mask']
    with torch.no_grad():
        original_outputs = model(
            input_ids=original_input_ids,
            attention_mask=original_attention_mask,
        ).logits  # [batch_size, seq_len, vocab_size]

    # 修改未来 token（例如将最后一个 token 替换为一个新的 token）
    modified_input_ids = original_input_ids.clone()
    modified_input_ids[:, -1] = random.randint(0, model.config.vocab_size - 1)
    with torch.no_grad():
        modified_outputs = model(
            input_ids=modified_input_ids,
            attention_mask=original_attention_mask,
        ).logits  # [batch_size, seq_len, vocab_size]

    # 比较原始输出和修改未来 token 后的输出在监测位置（例如 seq_len-2）是否相同
    monitor_position = -2  # 监测倒数第二个位置的 logits
    original_monitor_logits = original_outputs[:, monitor_position, :]  # [batch_size, vocab_size]
    modified_monitor_logits = modified_outputs[:, monitor_position, :]  # [batch_size, vocab_size]

    diff = torch.abs(original_monitor_logits - modified_monitor_logits)
    max_diff = diff.max().item()
    print(f'Max difference in monitor position logits after modifying future token: {max_diff:.6f}')
    assert max_diff < 1e-5, "Causal mask future leakage detected! The model's monitor position logits changed after modifying a future token, indicating that the causal mask may not be properly preventing future information leakage."


if __name__ == '__main__':
    test()
