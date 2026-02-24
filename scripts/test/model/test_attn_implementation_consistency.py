import sys
from pathlib import Path

import torch
from transformers.masking_utils import create_causal_mask

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.test.utils import set_seed, generate_random_inputs
from src.module.configuration import QiChatConfig
from src.module.model import QiChatAttention, QiChatRotaryPositionEmbedding


def main():
    use_seed = 36
    use_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_dtype = 'bfloat16'
    batch_size = 2
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
    attn_weights = QiChatAttention(
        config=module_config,
        layer_idx=0,
    ).state_dict()

    inputs = generate_random_inputs(
        batch_size=batch_size,
        seq_length=seq_len,
        d_model=module_config.d_model,
        pad_token_id=module_config.pad_token_id,
        dtype=module_config.dtype,
        device=use_device
    )

    rotary_pos_emb = QiChatRotaryPositionEmbedding(
        dim=module_config.d_kv,
        base=module_config.rope_theta,
        dtype=module_config.dtype,
    ).to(use_device).eval()

    cos_buf, sin_buf = rotary_pos_emb(
        position_ids=torch.arange(0, seq_len, device=use_device).unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
    )  # [1, seq_len, dim/2]

    with torch.no_grad():
        attn_implementation_hidden_states_map = {}
        for attn_implementation in ['eager', 'sdpa', 'flash_attention_2', 'flash_attention_3', 'flex_attention']:
            if use_device.type != 'cuda' and attn_implementation in ['flash_attention_2', 'flash_attention_3', 'flex_attention']:
                continue  # 仅在 CUDA 设备上测试 Flash Attention 和 Flex Attention
            module_config._attn_implementation = attn_implementation
            module = QiChatAttention(
                config=module_config,
                layer_idx=0,
            )
            module.load_state_dict(attn_weights)
            module = module.to(device=use_device, dtype=module_config.dtype).eval()

            hidden_states = inputs['hidden_states']
            attention_mask = create_causal_mask(
                config=module_config,
                input_embeds=hidden_states,
                attention_mask=inputs['attention_mask'],
                cache_position=torch.arange(0, seq_len, device=use_device),
                past_key_values=None,
            )
            hidden_states, attention_weights = module(
                hidden_states=hidden_states,
                mask=attention_mask,
                cos_buf=cos_buf,
                sin_buf=sin_buf,
            )
            attn_implementation_hidden_states_map[attn_implementation] = hidden_states[inputs['attention_mask']].detach()

    # 比较各注意力实现的输出是否一致
    # 误差字典
    dtype_tolerance_map = {
        torch.float16: {'rtol': 1e-2, 'atol': 1e-2},
        torch.bfloat16: {'rtol': 2e-2, 'atol': 2e-2},
        torch.float32: {'rtol': 1e-4, 'atol': 1e-5},
    }
    reference_hidden_states = attn_implementation_hidden_states_map.pop('eager')
    for impl, hidden_states in attn_implementation_hidden_states_map.items():
        max_diff = torch.max(torch.abs(reference_hidden_states - hidden_states)).item()
        print(f'[i] Comparing implementation {impl}, max absolute difference: {max_diff:.6f} ...', end=' ')
        try:
            torch.testing.assert_close(
                reference_hidden_states, hidden_states,
                **dtype_tolerance_map[module_config.dtype]
            )
        except AssertionError:
            print(f'Failed!')
        else:
            print('Passed.')


if __name__ == '__main__':
    main()
