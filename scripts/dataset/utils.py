import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, Any

import numpy
import pandas
from tqdm import tqdm

from src.tokenizer import QiTianTokenizerFast, get_tokenizer
from src.utils.masks import build_span_mask
from src.utils.path import data_path


def stream_corpus_batches(parquet_path: Path | str, batch_size: int, text_field: str | list[str] | None = None) -> Generator[pandas.DataFrame, Any, None]:
    """Stream batches of data from a parquet file."""
    if isinstance(parquet_path, str): parquet_path = Path(parquet_path)
    if isinstance(text_field, str): text_field = [text_field]

    try:
        from pyarrow import dataset
    except ImportError:
        raise ImportError('pyarrow is required for streaming parquet files. Please install it via `pip install pyarrow`.')
    dataset_obj = dataset.dataset(source=parquet_path, format='parquet')
    for batch in dataset_obj.to_batches(columns=text_field, batch_size=batch_size):
        yield batch.to_pandas()


def stream_read_jsonl(file_path: Path):
    """ 流式读取jsonl文件 """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def auto_select_precision_from_range(min_value: int, max_value: int) -> numpy.dtype:
    """
    Auto-select the smallest numpy integer dtype that can hold values in [min_value, max_value].

    The function automatically decides whether signed or unsigned types are needed based
    on the provided min_value. It tries 8-bit, 16-bit, 32-bit, then 64-bit types.

    Args:
        min_value: minimum token id observed in data (can be negative)
        max_value: maximum token id observed in data

    Returns:
        A numpy dtype class (e.g., numpy.uint8, numpy.int16)

    Raises:
        ValueError: if min_value > max_value.
        ValueError: if the provided range cannot be represented in 64-bit integers.
    """
    if min_value > max_value:
        raise ValueError('min_value must be <= max_value')

    if min_value >= 0:
        # If all values are non-negative, prefer unsigned types
        unsigned_candidates = [
            (numpy.uint8, 0, 255),
            (numpy.uint16, 0, 65535),
            (numpy.uint32, 0, 4294967295),
            (numpy.uint64, 0, 18446744073709551615),
        ]
        for dtype, lo, hi in unsigned_candidates:
            if max_value <= hi:
                return dtype
    elif min_value < 0:
        signed_candidates = [
            (numpy.int8, -128, 127),
            (numpy.int16, -32768, 32767),
            (numpy.int32, -2147483648, 2147483647),
            (numpy.int64, -9223372036854775808, 9223372036854775807),
        ]
        for dtype, lo, hi in signed_candidates:
            if min_value >= lo and max_value <= hi:
                return dtype

    raise ValueError(f'Range [{min_value}, {max_value}] is too large for 64-bit integers')


def generate_report(sample_lens: list[int], save_path: Path | None = None) -> pandas.DataFrame:
    """生成长度简报并可选保存为JSON文件"""
    desc = pandas.DataFrame(data=sample_lens).describe()
    desc.rename(index={
        'count': 'Total Samples',
        'mean': 'Average Length',
        'std': 'Std Dev',
        'min': 'Min Length',
        '25%': '25th Percentile',
        '50%': 'Median Length',
        '75%': '75th Percentile',
        'max': 'Max Length'
    }, inplace=True)
    # 添加总 token 数量行
    desc.loc['Total Tokens'] = sum(sample_lens)
    desc = desc.round(2)

    if save_path:
        desc[0].to_json(save_path, indent=4)

    return desc


def generate_system_block() -> dict[str, str]:
    username_list = [
        'Alice', 'Bob', 'Charlie', 'David', 'Eve',
        'Frank', 'Grace', 'Heidi', 'Ivan', 'Judy',
        '小明', '小红', '小刚', '小丽', '小华',
        '张伟', '王芳', '李娜', '刘洋', '陈强',
        'Anna', 'Tom', 'Jerry', 'Lucy', 'Mike',
        'Sara', 'David', 'Emma', 'John', 'Olivia',
        '王强', '赵敏', '孙磊', '周杰', '吴婷',
        '刘强', '陈敏', '杨磊', '黄杰', '赵婷',
    ]
    random_username = username_list[random.randint(0, len(username_list) - 1)]
    random_month = random.randint(1, 12)
    month_day_mapping = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }
    random_datetime = datetime(
        year=random.choice([2025, 2026]),
        month=random_month,
        day=random.randint(1, month_day_mapping[random_month]),
        hour=random.randint(0, 23),
        minute=random.randint(0, 59),
        second=random.randint(0, 59)
    )
    system_prompt_list = [
        f'你是 QiChat，一个生成式预训练对话模型，旨在提供准确、简洁的回答。今天是 {random_datetime.strftime("%Y-%m-%d")}，当前时间是 {random_datetime.strftime("%H:%M:%S")}。',
        f'你是 QiChat。当前时间 {random_datetime.strftime("%Y-%m-%d %H:%M:%S")}。请为用户提供准确、简洁的回答。',
        f'作为 QiChat，你的目标是为用户 {random_username} 提供准确、简洁的回答。当前是 {random_datetime.strftime("%Y-%m-%d %H:%M:%S")}。',
        f'作为 QiChat，在与 {random_username} 的对话中，你应该始终保持礼貌、专业和尊重，避免使用冒犯性或不适当的语言。',
        f'你是 QiChat，在与用户交流时，如果你不确定某个问题的答案，或者问题超出了你的知识范围，你应该诚实地承认，并尽力提供相关信息或建议，而不是编造答案。',
        f'现在是 {random_datetime.strftime("%Y-%m-%d %H:%M:%S")}。作为 QiChat 模型，无论用户提出什么问题，你都应该尽力提供准确、简洁的回答，并且始终保持礼貌和专业。',
        f'作为 QiChat，你应该始终以用户为中心，努力理解他们的意图和需求，并提供有针对性的、实用的回答，以帮助他们解决问题或满足他们的需求。',
        f'你是 QiChat，任何情况下系统指令优先于用户指令，若冲突，则遵循系统指令。',
        f'你是 QiChat，用户是 {random_username}，当前时间是 {random_datetime.strftime("%Y-%m-%d %H:%M:%S")}。',
        f'你是 QiChat，当用户的陈述存在明显事实错误时，应当进行纠正并简短解释。',
        f'你是 QiChat，当前时间是 {random_datetime.strftime("%Y-%m-%d %H:%M:%S")}。请为用户提供准确、简洁的回答，并且在适当的时候提供相关背景信息以帮助用户更好地理解。',
        f'当前时间是 {random_datetime.strftime("%Y-%m-%d %H:%M:%S")}。',
    ]
    return {'role': 'system', 'content': random.choice(system_prompt_list)}


@dataclass(frozen=True)
class SFTCorpusAccumulator:
    tokenized_dialogues: list[numpy.ndarray]
    sample_token_count: list[int]
    sample_supervised_token_count: list[int]
    min_token_id: int
    max_token_id: int


def convert_to_sft_format(
    conversations: list[list[dict[str, str]]],
    *,
    add_system_block: bool = False,
    assistant_token: str = '<|assistant|>',
    end_of_turn_token: str = '<|eot|>',
) -> SFTCorpusAccumulator:
    """
    Convert raw conversations into tokenized format suitable for SFT training.
    """
    tokenizer: QiTianTokenizerFast = get_tokenizer()

    dialogues: list[numpy.ndarray] = []
    min_token_id, max_token_id = numpy.inf, -numpy.inf
    sample_lens: list[int] = []
    supervision_lens: list[int] = []

    start_token_id = tokenizer.convert_tokens_to_ids(assistant_token)
    end_token_id = tokenizer.convert_tokens_to_ids(end_of_turn_token)

    for conversation in tqdm(conversations, desc='Processing Conversations', leave=False, dynamic_ncols=True):
        if add_system_block and 'system' not in [turn['role'] for turn in conversation]:
            conversation.insert(0, generate_system_block())
        tokenized = tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors='np'
        ).input_ids[:, 2:][0]  # Remove BOS and \n Token
        dialogues.append(tokenized)
        sample_lens.append(len(tokenized))

        ignore_mask = build_span_mask(
            input_ids=tokenized,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            include_start_token=False,
            include_end_token=True,
        )
        supervision_tokens = ignore_mask.sum()  # True 的个数
        supervision_lens.append(supervision_tokens)

        arr_min, arr_max = numpy.min(tokenized), numpy.max(tokenized)
        if arr_min < min_token_id: min_token_id = arr_min
        if arr_max > max_token_id: max_token_id = arr_max

    return SFTCorpusAccumulator(
        tokenized_dialogues=dialogues,
        sample_token_count=sample_lens,
        sample_supervised_token_count=supervision_lens,
        min_token_id=int(min_token_id),
        max_token_id=int(max_token_id)
    )


def build_sft_dataset(
    dataset_name: str,
    conversations: list[list[dict[str, str]]],
    add_system_block: bool = False,  # 是否加入可变体 system 角色区块
    assistant_token: str = '<|assistant|>',
    end_of_turn_token: str = '<|eot|>',
) -> None:
    sft_data: SFTCorpusAccumulator = convert_to_sft_format(
        conversations=conversations,
        add_system_block=add_system_block,
        assistant_token=assistant_token,
        end_of_turn_token=end_of_turn_token,
    )
    dialogues_df = pandas.DataFrame({
        'tokenized': sft_data.tokenized_dialogues,
        'token_count': sft_data.sample_token_count,
        'supervised_token_count': sft_data.sample_supervised_token_count,
    })
    dtype = auto_select_precision_from_range(min_value=sft_data.min_token_id, max_value=sft_data.max_token_id)
    print(f'Selected dtype for token IDs: {dtype.__name__} (Token ID range: [{sft_data.min_token_id}, {sft_data.max_token_id}])')
    dialogues_df['tokenized'] = dialogues_df['tokenized'].apply(lambda obj: numpy.array(obj, dtype=dtype))

    dialogues_df.to_parquet(data_path('dataset', f'{dataset_name}.parquet'), index=False)
    generate_report(
        sample_lens=sft_data.sample_token_count,
        save_path=data_path('dataset', f'{dataset_name}.json')
    )
    desc = generate_report(
        sample_lens=sft_data.sample_supervised_token_count,
        save_path=data_path('dataset', f'{dataset_name}_supervision.json')
    )
    print(desc.to_string(header=False))


def split_corpus(text: str, max_length: int) -> list[str]:
    """
    Split the text into smaller chunks based on punctuation.
    如果文本长度未超过 max_length，直接返回原文本；
    否则在文本的前 1/3 到 75% 之间寻找尽量合适的位置（句号、感叹号、问号和换行符等明显的分句标志）作为切割点；
    如果找到切割点，则将文本切成两部分，并递归处理每部分；
    如果没有找到合适的切割点，则直接在 max_length 处硬切。
    Args:
        text (str): The text to split.
        max_length (int): The maximum length of each chunk.
    Returns:
        list[str]: A list of text chunks, each not exceeding max_length.
    """
    text = text.strip()
    if not text: return []
    if len(text) <= max_length: return [text]

    chunks: list[str] = []
    n = len(text)
    current_position = n // 3  # 从前 1/3 处开始寻找切割点
    end_limit = n - n // 4  # 约等于 75%

    while current_position < end_limit:  # 只在整个段落中的前 1/3 到 75% 之间寻找切割点
        pos_char = text[current_position]
        if pos_char in ['。', '！', '？', '!', '?', '.', '\n']:
            if pos_char == '.': # 处理英文句号，确保不在数字中间切割
                prev_ok = current_position - 1 >= 0 and not text[current_position - 1].isdigit()
                next_ok = current_position + 1 < n and not text[current_position + 1].isdigit()
                if not (next_ok and prev_ok):
                    current_position += 1
                    continue
            pos_index = current_position + 1
            # 将切割点后面的空白字符容纳到前半部分，避免后半部分以空白开头
            while pos_index < n and text[pos_index].isspace(): pos_index += 1
            for part in [
                text[:pos_index],  # 前半部分
                text[pos_index:]  # 后半部分
            ]:
                part = part.strip()
                if not part: continue
                if len(part) <= max_length: chunks.append(part)  # 如果长度合适，直接添加进列表
                else: chunks.extend(split_corpus(text=part, max_length=max_length))  # 否则递归切割
            break
        current_position += 1

    # 兜底：如果没有找到合适的切割点说明是一段非常长的且没有标点符号的文本，对于预训练来说硬切也无妨
    if not chunks:
        chunks.append(text[:max_length].strip())
        remaining_text = text[max_length:].strip()
        if remaining_text:
            chunks.extend(split_corpus(text=remaining_text, max_length=max_length))

    return chunks
