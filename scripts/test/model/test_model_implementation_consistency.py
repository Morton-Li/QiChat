import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.test.utils import set_seed, generate_random_inputs
from src.module import QiChatConfig, QiChatForCausalLM


def main():
    use_seed = 36
    use_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_dtype = 'bfloat16'
    seq_len = 16

    print(f'Using device: {use_device.type}, dtype: {use_dtype}, seq_len: {seq_len}')

    set_seed(seed=use_seed)

    module_config = QiChatConfig(
        d_model=64,
        num_hidden_layers=2,
        n_heads=4,
        d_kv=16,
        n_kv_heads=2,
        d_ff=256,
        attn_implementation='eager',
        dtype=use_dtype,
    )  # 2.17 M
    model_weights = QiChatForCausalLM(config=module_config).state_dict()

    inputs = generate_random_inputs(
        batch_size=2,
        seq_length=seq_len,
        vocab_size=module_config.vocab_size,
        pad_token_id=module_config.pad_token_id,
        dtype=module_config.dtype,
        device=use_device,
    )

    with torch.no_grad():
        attn_implementation_logits_map = {}
        for attn_implementation in ['eager', 'sdpa', 'flash_attention_2', 'flash_attention_3', 'flex_attention']:
            if use_device.type != 'cuda' and attn_implementation in ['flash_attention_2', 'flash_attention_3', 'flex_attention']:
                continue  # 仅在 CUDA 设备上测试 Flash Attention 和 Flex Attention
            module_config._attn_implementation = attn_implementation

            model = QiChatForCausalLM(config=module_config).to(device=use_device, dtype=module_config.dtype)
            model.load_state_dict(model_weights)
            model.eval()
            out = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                use_cache=False,
            )
            attn_implementation_logits_map[attn_implementation] = out.logits

    # 比较各实现的输出是否一致
    # 误差字典
    dtype_tolerance_map = {
        torch.float16: {'rtol': 1e-2, 'atol': 1e-2},
        torch.bfloat16: {'rtol': 2e-2, 'atol': 2e-2},
        torch.float32: {'rtol': 1e-4, 'atol': 1e-5},
    }
    reference_logits = attn_implementation_logits_map.pop('eager')
    for impl, logits in attn_implementation_logits_map.items():
        max_diff = torch.max(torch.abs(reference_logits - logits)).item()
        print(f'[i] Comparing implementation {impl}, max absolute difference: {max_diff:.6f} ...', end=' ')
        try:
            torch.testing.assert_close(
                reference_logits, logits,
                **dtype_tolerance_map[module_config.dtype]
            )
        except AssertionError: print(f'Failed!')
        else: print('Passed.')


if __name__ == '__main__':
    main()
