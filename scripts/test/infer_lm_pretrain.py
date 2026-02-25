"""
QiChat 模型推理脚本
Usage:
    python scripts/infer_lm_pretrain.py [--seed SEED] [--device DEVICE] [--greedy_decoding]
"""
import argparse
import sys
from pathlib import Path
from typing import Literal

import torch
from transformers import GenerationConfig

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.module.model import QiChatForCausalLM
from src.tokenizer import QiTianTokenizerFast, get_tokenizer
from scripts.test.utils import set_seed, get_device, init_model, load_model_checkpoint


def main(
    device: Literal['auto', 'cpu', 'cuda', 'mps'] = 'auto',
    seed: int = 36,
    greedy_decoding: bool = True,
):
    """ Main inference function. """
    set_seed(seed=seed)

    use_device: torch.device = get_device(device=device)
    print(f'Using device: {use_device.type}')

    tokenizer: QiTianTokenizerFast = get_tokenizer()
    model: QiChatForCausalLM = init_model(
        param_size='0.3B',
        attn_implementation='sdpa',
        dtype='bfloat16',
        additional_model_config_kwargs=None,
        checkpoint=load_model_checkpoint(map_location=use_device)['state']['weights'],
        use_device=use_device,
    )
    if model.get_input_embeddings().weight.size(0) != len(tokenizer):
        print(f'Token embeddings size ({model.get_input_embeddings().weight.size(0)}) does not match tokenizer size ({len(tokenizer)}). Resizing token embeddings.')
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    eos_token_ids = [tokenizer.convert_tokens_to_ids('<|eot|>'), tokenizer.eos_token_id]
    sample_inputs = [
        '量子纠缠是一种发生在多个粒子之间的量子关联现象，其特点是',
        '要在 Linux 上查看当前系统的内核版本，可以使用如下命令：',
        '明清时期的海上贸易在东亚格局中扮演了重要角色，其中',
        '通货膨胀通常指总体物价水平持续上涨的现象。在宏观经济分析中，',
        '雨停之后，巷子里仍残留着潮湿的气息。他站在屋檐下，',
        '“你觉得明天会下雨吗？”他抬头看了看天空，说：',
        'The French Revolution was a period of political and social upheaval in France that began',
        'A neural network learns by adjusting its parameters to minimize a loss function. In practice,',
        'During the Industrial Revolution, advances in manufacturing and transportation led to',
        'Inflation refers to a sustained increase in the general price level of goods and services. One common measure is',
        'Let X be a random variable with finite variance. Then the law of large numbers states that',
        'When the train finally arrived, she realized she had been waiting for hours, and'
    ]

    # 贪婪解码或采样解码
    generation_config_kwargs = {'do_sample': False}
    if greedy_decoding: print(f'Using greedy decoding for inference.')
    else:
        print(f'Using sampling decoding for inference.')
        generation_config_kwargs.update({
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.2,
            'no_repeat_ngram_size': 3,
        })

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_ids,
        use_cache=True,
        max_new_tokens=64,
        return_dict_in_generate=True,
        output_logits=True,
        **generation_config_kwargs
    )

    print(f'Starting inference ...')

    inputs = tokenizer(sample_inputs, return_tensors='pt', padding=True, padding_side='left', add_special_tokens=False)  # add_special_tokens=False 为去除 bos 和 eos
    outputs = model.generate(**inputs.to(use_device), generation_config=generation_config)
    sequences = outputs.sequences  # [batch_size, input_len + gen_len]
    logits = outputs.logits  # tuple of (tensor of shape [batch_size, vocab_size]) with length gen_len

    outputs_trimmed: list[list[int]] = sequences[:, inputs.input_ids.size(1):].detach().to("cpu").tolist()  # 去掉 input 部分 [batch_size, gen_len]

    stop_set = {tokenizer.convert_tokens_to_ids("<|eot|>"), tokenizer.eos_token_id}
    # 在生成序列中找到第一个停止符位置并截断
    for i in range(len(outputs_trimmed)):
        for j in range(len(outputs_trimmed[i])):
            if outputs_trimmed[i][j] in stop_set:
                outputs_trimmed[i] = outputs_trimmed[i][:j + 1]  # 保留停止符
                break

    generated = tokenizer.batch_decode(outputs_trimmed, skip_special_tokens=False)

    print(f'-' * 36)
    for i, (input_text, generated_text, generated_ids) in enumerate(zip(sample_inputs, generated, outputs_trimmed)):
        print(f'Input: {input_text}')
        print(f'Generation: {generated_text}')
        print(f'Stopped by EOS: {generated_ids[-1] in eos_token_ids}')
        print(f'-' * 36)

    print(f'Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QiChat 模型预训练推理脚本')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'], help='设备选择，默认自动选择设备。')
    parser.add_argument('--seed', type=int, default=36, help='随机种子，默认为 36。')
    parser.add_argument('--greedy_decoding', action='store_true', help='是否使用贪婪解码，默认为采样解码。')
    args = parser.parse_args()

    main(
        device=args.device,
        seed=args.seed,
        greedy_decoding=args.greedy_decoding,
    )
