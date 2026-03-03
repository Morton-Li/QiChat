#!/usr/bin/env python3
"""
Evaluate the model on the MMLU benchmark using the lm-eval framework.

Usage:
    python scripts/eval/mmlu.py [--param_size PARAM_SIZE] [--attn_implementation ATTENTION_IMPLEMENTATION] [--dtype DTYPE] [--use_device USE_DEVICE]

Requires:
    - "lm-eval[hf]"
"""
import argparse
import sys
from pathlib import Path
from typing import Literal

from lm_eval.evaluator import simple_evaluate

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from scripts.eval.utils import init_eval_model


def main(
    param_size: Literal['Tiny', '73M', '0.3B', '0.6B', '1.3B', '3.2B'] = 'Tiny',
    attn_implementation: Literal['eager', 'memory_efficient', 'sdpa', 'flash_attention_2', 'flash_attention_3'] = 'flash_attention_2',
    dtype: Literal['float16', 'bfloat16', 'float32'] = 'bfloat16',
    use_device: Literal['cuda', 'mps', 'cpu'] = 'cuda',
):
    print(f'Running MMLU evaluation...')

    results = simple_evaluate(
        model=init_eval_model(
            param_size=param_size,
            attn_implementation=attn_implementation,
            dtype=dtype,
            use_device=use_device,
        ),
        tasks=['mmlu'],
        batch_size=16,
        apply_chat_template=False
    )
    print(f'results: {results}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model on MMLU benchmark.')
    parser.add_argument(
        '--param_size',
        type=str,
        default='Tiny',
        choices=['Tiny', '73M', '0.3B', '0.6B', '1.3B', '3.2B'],
        help='Model parameter size, default is Tiny.',
    )
    parser.add_argument(
        '--attn_implementation',
        type=str,
        default='flash_attention_2',
        choices=['eager', 'memory_efficient', 'sdpa', 'flash_attention_2', 'flash_attention_3'],
        help='Attention implementation to use, default is flash_attention_2.',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='bfloat16',
        choices=['float16', 'bfloat16', 'float32'],
        help='Data type for model weights, default is bfloat16.',
    )
    parser.add_argument(
        '--use_device',
        type=str,
        default='cuda',
        choices=['cuda', 'mps', 'cpu'],
        help='Device to run the evaluation on, default is cuda.',
    )
    args = parser.parse_args()

    main(
        param_size=args.param_size,
        attn_implementation=args.attn_implementation,
        dtype=args.dtype,
        use_device=args.use_device,
    )
