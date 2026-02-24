#!/usr/bin/env python3
"""
与 QiChat 模型进行对话

Usage:
    python scripts/chat_with_model.py
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Literal

import torch
from transformers import GenerationConfig, TextIteratorStreamer

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.module import QiChatForCausalLM
from src.tokenizer import QiTianTokenizerFast, get_tokenizer
from scripts.test.utils import set_seed, get_device, init_model, load_model_checkpoint


def generate_input_ids(
    tokenizer: QiTianTokenizerFast,
    conversations: list[dict],
    add_system_prompt: bool = True,
) -> torch.Tensor:
    """ Generate input_ids from conversations. """
    # 插入系统提示
    if add_system_prompt and 'system' not in [turn['role'] for turn in conversations]:
        system_prompt = {
            'role': 'system',
            'content': f'你是QiChat，现在是 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}。请根据用户提出的问题进行准确、简洁的回答，如果你不确定某个问题的答案，可以直接说“我不知道”。'
        }
        conversations.insert(0, system_prompt)
    input_ids = tokenizer.encode(
        text=tokenizer.apply_chat_template(
            conversation=conversations,
            add_generation_prompt=True,
            tokenize=False,
        ),
        return_tensors='pt',
        add_special_tokens=False
    )[:, 2:]  # 剔除 bos 和 换行符
    return input_ids


def main(
    model_size: Literal['Tiny', '73M', '0.3B', '0.6B', '1.3B', '3.2B'],
    seed: int = 36,
    greedy_decoding: bool = False,
    use_device: Literal['auto', 'cuda', 'mps', 'cpu'] = 'auto',
    attn_implementation: Literal['flash_attention_2', 'sdpa', 'memory_efficient'] = 'sdpa',
):
    """ Main inference function. """
    set_seed(seed=seed)

    use_device: torch.device = get_device(device=use_device)
    print(f'Using device: {use_device.type}')

    tokenizer: QiTianTokenizerFast = get_tokenizer()
    model: QiChatForCausalLM = init_model(
        param_size=model_size,
        attn_implementation=attn_implementation,
        dtype='bfloat16',
        additional_model_config_kwargs=None,
        checkpoint=load_model_checkpoint(map_location=use_device)['state']['weights'],
        use_device=use_device,
    ).eval()
    if model.get_input_embeddings().weight.size(0) != len(tokenizer):
        print(f'Error: Token embeddings size ({model.get_input_embeddings().weight.size(0)}) does not match tokenizer size ({len(tokenizer)}).')
        sys.exit(1)

    # 贪婪解码或采样解码
    generation_config_kwargs = {'do_sample': False}
    if greedy_decoding:
        print(f'Using greedy decoding for inference.')
    else:
        print(f'Using sampling decoding for inference.')
        generation_config_kwargs.update({
            'do_sample': True,
            'temperature': 0.8,
            'top_p': 0.92,
            'repetition_penalty': 1.08,
            'no_repeat_ngram_size': 2,
        })
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[tokenizer.convert_tokens_to_ids('<|eot|>'), tokenizer.eos_token_id],
        use_cache=True,
        max_new_tokens=512,
        **generation_config_kwargs
    )
    conversations = []

    try:
        while True:
            user_input = input('User: ')
            conversations.append({'role': 'user', 'content': user_input})
            input_ids = generate_input_ids(
                tokenizer=tokenizer,
                conversations=conversations,
                add_system_prompt=False,
            )

            while input_ids.shape[-1] > model.config.max_position_embeddings:
                # 超长对话截断
                conversations.pop(0)
                input_ids = generate_input_ids(
                    tokenizer=tokenizer,
                    conversations=conversations,
                    add_system_prompt=False,
                )

            # 推理线程
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generate_thread = Thread(
                target=model.generate,
                kwargs={
                    'input_ids': input_ids.to(use_device),
                    'generation_config': generation_config,
                    'streamer': streamer,
                }
            )
            generate_thread.start()

            generate_text = ''
            print('Assistant: ', end='', flush=True)
            for new_text in streamer:
                print(new_text, end='', flush=True)
                generate_text += new_text
            print()
            conversations.append({'role': 'assistant', 'content': generate_text})

            generate_thread.join()
    except KeyboardInterrupt:
        print('\n结束对话。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='与 QiChat 模型进行对话')
    parser.add_argument('--device', type=str, default='auto', help='选择设备: auto, cuda, mps, cpu')
    args = parser.parse_args()

    main(
        model_size='0.3B',
        use_device=args.device,
    )
