"""
Script to build the OASST2 dialogue dataset from the source corpus.

Usage:
    python scripts/dataset/sft/build_oasst2_dialogue_dataset.py
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.utils.path import data_path
from scripts.dataset.utils import build_sft_dataset


def extract_dialogue(node, current_path=None):
    """ 递归提取对话信息 """
    if current_path is None: current_path = []
    if node["role"] == 'prompter': node["role"] = 'user'

    # 当前节点加入路径
    current_path = current_path + [{
        "role": node["role"].strip(),
        "content": node["text"].strip()
    }]

    # 如果没有子节点，则这是一个完整路径
    if not node.get("replies"):
        yield current_path
    else:
        # 否则递归展开每个子回复
        for reply in node["replies"]:
            yield from extract_dialogue(reply, current_path)


def read_corpus(file_path: Path, language: list[str]) -> list[list[dict[str, str]]]:
    """ 读取语料库文件，返回包含所有对话的列表 """
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus = [json.loads(line)['prompt'] for line in f]
    dataset: list[list[dict[str, str]]] = []
    for prompt in corpus:
        if prompt['lang'] not in language: continue

        messages: list[list[dict[str, str]]] = [
            message for message in extract_dialogue(prompt)
        ]
        dataset.extend(messages)

    return dataset


def main():
    # Open Assistant Conversations Dataset Release 2 (OASST2)
    print("Hello, OASST2!")

    for lang in ['en', 'zh']:
        print(f'Processing {lang} dialogues...')
        conversations = [
            messages for messages in read_corpus(
                file_path=data_path('tokenizer', 'source_corpus', '2023-11-05_oasst2_ready.trees.jsonl'),
                language=[lang]
            )
        ]
        print(f'Extracted {len(conversations)} dialogues from the corpus.')

        build_sft_dataset(
            dataset_name=f'OASST2_{lang}',
            conversations=conversations,
            add_system_block=False,
            assistant_token='<|assistant|>',
            end_of_turn_token='<|eot|>',
        )
    print('Dialogue dataset built successfully.')


if __name__ == '__main__':
    main()
