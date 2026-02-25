"""
QiChat 模型推理脚本
Usage:
    python scripts/infer_chat.py [--seed SEED] [--device DEVICE] [--greedy_decoding] [--debug]
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Literal

import torch
from transformers import GenerationConfig, TextIteratorStreamer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import get_config
from src.module.model import QiChatForCausalLM
from src.tokenizer import QiTianTokenizerFast, get_tokenizer
from src.utils.type import DotDict
from scripts.test.utils import set_seed, get_device, init_model, load_model_checkpoint


def main(
    max_length: int = 512,
    seed: int = 36,
    use_device: Literal['auto', 'cuda', 'mps', 'cpu'] = 'auto',
    greedy_decoding: bool = True,
    debug: bool = False
):
    """ Main inference function. """
    print(f'Random seed set to: {seed}')
    set_seed(seed=seed)

    use_device: torch.device = get_device(device=use_device)
    print(f'Using device: {use_device.type}')

    tokenizer: QiTianTokenizerFast = get_tokenizer()
    config: DotDict = get_config()
    model: QiChatForCausalLM = init_model(
        param_size=config.model.param_size,
        attn_implementation=config.model.attn_implementation,
        dtype=config.model.dtype,
        additional_model_config_kwargs=config.model.model_config_kwargs,
        checkpoint=load_model_checkpoint(map_location=use_device)['state']['weights'],
        use_device=use_device,
    )
    if model.get_input_embeddings().weight.size(0) != len(tokenizer):
        print(f'Token embeddings size ({model.get_input_embeddings().weight.size(0)}) does not match tokenizer size ({len(tokenizer)}). Resizing token embeddings.')
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    eos_token_ids = [tokenizer.convert_tokens_to_ids('<|eot|>'), tokenizer.eos_token_id]
    samples = [
        '相对论的作者是谁？',
        'Who is the author of the theory of relativity?',
        '孔子是什么时期的人？',
        '美国共有多少个州？',
        'How many states are there in the United States?',
        '中国的全称是什么？',
        '什么是北回归线？',
        '32 加 45 等于多少？',
    ]
    # system_prompt = f'你是QiChat，现在是 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}。请根据用户提出的问题进行准确、简洁的回答，如果你不确定某个问题的答案，可以直接说“我不知道”。'

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
        max_new_tokens=max_length,
        **generation_config_kwargs
    )

    print(f'Starting inference ...')
    input_chat_template = tokenizer.apply_chat_template(
        conversation=[
            [
                # {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': sample}
            ] for sample in samples
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

    print(f'-' * 36)
    if debug:
        # 调试模式，直接批生成
        inputs = tokenizer(input_chat_template, return_tensors='pt', padding=True, padding_side='left', add_special_tokens=False)
        # Remove BOS and \n Token
        cur, nxt = inputs.input_ids[:, :-1], inputs.input_ids[:, 1:]
        mask = (cur == 0) & (nxt == 218)
        inputs.input_ids[:, :-1][mask] = tokenizer.pad_token_id
        inputs.input_ids[:, 1:][mask] = tokenizer.pad_token_id
        inputs.attention_mask = inputs.input_ids != tokenizer.pad_token_id

        outputs = model.generate(**inputs.to(use_device), generation_config=generation_config)

        outputs_trimmed: list[list[int]] = outputs[:, inputs.input_ids.size(1):].detach().to("cpu").tolist()  # 去掉 input 部分 [batch_size, gen_len]

        stop_set = {tokenizer.convert_tokens_to_ids("<|eot|>"), tokenizer.eos_token_id}
        # 在生成序列中找到第一个停止符位置并截断
        for i in range(len(outputs_trimmed)):
            for j in range(len(outputs_trimmed[i])):
                if outputs_trimmed[i][j] in stop_set:
                    outputs_trimmed[i] = outputs_trimmed[i][:j + 1]  # 保留停止符
                    break

        generated = tokenizer.batch_decode(outputs_trimmed, skip_special_tokens=False)
        for i, (input_text, generated_text, generated_ids) in enumerate(zip(samples, generated, outputs_trimmed)):
            print(f'User: {input_text}')
            print(f'Assistant: {generated_text}')
            print(f'Stopped by EOS: {generated_ids[-1] in eos_token_ids}')
            print(f'-' * 36)
    else:
        for i, (input_text, chat_template) in enumerate(zip(samples, input_chat_template)):
            print(f'User: {input_text}')
            print(f'Assistant: ', end='', flush=True)
            # 流式生成模式
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            # 推理线程
            generate_thread = Thread(
                target=model.generate,
                kwargs={
                    'input_ids': tokenizer.encode(chat_template, return_tensors='pt', add_special_tokens=False)[:, 2:].to(use_device),  # 剔除 bos 和 换行符
                    'generation_config': generation_config,
                    'streamer': streamer,
                }
            )
            generate_thread.start()

            # 实时打印输出
            for new_text in streamer:
                print(new_text, end='', flush=True)
            print()

            generate_thread.join()

            print(f'-' * 36)

    print(f'Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QiChat 模型对话推理脚本')
    parser.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='生成文本的最大长度，默认为 128。',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=36,
        help='随机种子，默认为 36。',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='指定运行设备，默认为 auto 自动选择。',
    )
    parser.add_argument(
        '--greedy_decoding',
        action='store_true',
        help='是否使用贪婪解码，默认为采样解码。',
    )
    parser.add_argument(
        '--debug',
        type=bool,
        default=False,
        help='启用调试模式，关闭流式生成并打印更多信息。',
    )
    args = parser.parse_args()

    main(
        max_length=args.max_length,
        seed=args.seed,
        use_device=args.device,
        greedy_decoding=args.greedy_decoding,
        debug=args.debug
    )
