import random
import sys
from pathlib import Path
from typing import Optional, Literal

import numpy
import pandas

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.masks import build_span_mask
from src.utils.path import data_path
from scripts.dataset.utils import stream_corpus_batches, auto_select_precision_from_range, generate_report


def process_corpus_file(
    parquet_path: Path,
    dataset_type: Literal['pretrain', 'sft'] = 'sft',
    filter_rate: float = 0.16,
    len_range: tuple[int, int] | None = None,
    dtype: Optional[numpy.dtype] = None,
):
    """
    过滤样本并生成简报。
    Args:
        parquet_path (Path): 输入的 Parquet 文件路径。
        dataset_type (Literal['pretrain', 'sft'] = 'sft'): 数据集类型，'pretrain' 表示预训练数据集，'sft' 表示微调数据集。根据类型选择不同的过滤策略。
        filter_rate (float): 随机过滤样本的比例，范围为 0 到 1 之间，例如 0.16 表示过滤掉 16% 的样本。
        len_range (tuple[int, int] | None): 样本长度范围，格式为 (min_len, max_len)。如果为 None，则不进行长度过滤。
        dtype (Optional[numpy.dtype]): 指定用于存储 token ID 的数据类型。如果为 None，则自动选择合适的数据类型。
    生成:
        filtered_dataset.parquet: 过滤后的样本数据集。
        filtered_dataset.json: 过滤后样本的简报。
    """
    print('-' * 50)
    print(f'Processing corpus file: {parquet_path.name} ...')
    if len_range is not None: print(f'Length filter: {len_range[0]} <= sample_len <= {len_range[1]}')
    if filter_rate > 0: print(f'Random filter rate: {filter_rate * 100:.2f}%')

    dataset: list[numpy.ndarray[int]] = []
    sample_lens: list[int] = []
    for batch in stream_corpus_batches(parquet_path=parquet_path, batch_size=16000, text_field='tokenized'):
        print('.', end='', flush=True)
        for sample in batch.stack().dropna().reset_index(drop=True):
            # sample 为 numpy.ndarray
            if dataset_type == 'pretrain':
                sample_len = len(sample)
            elif dataset_type == 'sft':
                sample_len = build_span_mask(
                    input_ids=sample,
                    start_token_id=7,  # 7 是 <|assistant|> 的 token ID
                    end_token_id=2,  # 2 是 <|eot|> 的 token ID
                    include_start_token=False,
                    include_end_token=True,
                ).sum()  # 计算监督信号的 token 数量
            else: raise ValueError(f'Unsupported dataset type: {dataset_type}')
            # 过滤长度
            if len_range is not None:
                min_len, max_len = len_range
                if max_len < min_len: min_len, max_len = max_len, min_len
                if sample_len < min_len or sample_len > max_len: continue

            # 随机过滤掉 filter_rate 比例的样本
            if random.random() < filter_rate: continue

            dataset.append(sample)
            sample_lens.append(sample_len)

    print()
    print(f'Total processed samples: {len(dataset)}, Total Tokens: {sum(sample_lens)}')

    print('-' * 50)
    # 请用户确认是否继续保存过滤后的数据
    confirm = input('Continue to save the filtered dataset? (y/n): ').strip().lower()
    if confirm != 'y':
        print('Aborting saving filtered dataset.')
        return

    dialogues_df = pandas.DataFrame(data={'tokenized': dataset})

    if dtype is not None:
        min_token_id, max_token_id = numpy.inf, -numpy.inf
        for arr in dialogues_df['tokenized']:
            arr_min, arr_max = numpy.min(arr), numpy.max(arr)
            if arr_min < min_token_id: min_token_id = arr_min
            if arr_max > max_token_id: max_token_id = arr_max
        min_token_id, max_token_id = int(min_token_id), int(max_token_id)
        dtype = auto_select_precision_from_range(min_token_id, max_token_id)
        print(f'Selected dtype for token IDs: {dtype.__name__} (range: [{min_token_id}, {max_token_id}])')
    else:
        print(f'Selected dtype for token IDs: {dtype.__name__}')
    dialogues_df['tokenized'] = dialogues_df['tokenized'].apply(lambda obj: numpy.array(obj, dtype=dtype))
    parquet_path = data_path('dataset', f'{parquet_path.stem}_filtered.parquet')
    parquet_path.unlink(missing_ok=True)
    dialogues_df.to_parquet(parquet_path, index=False)

    # 简报
    report_path = data_path('dataset', f'{parquet_path.stem}.json')
    report_path.unlink(missing_ok=True)
    if dataset_type == 'pretrain':
        desc = generate_report(
            sample_lens=sample_lens,
            save_path=report_path
        )
    elif dataset_type == 'sft':
        generate_report(
            sample_lens=dialogues_df['tokenized'].apply(len).tolist(),
            save_path=report_path,
        )
        supervision_report_path = data_path('dataset', f'{parquet_path.stem}_supervision.json')
        supervision_report_path.unlink(missing_ok=True)
        desc = generate_report(
            sample_lens=sample_lens,
            save_path=supervision_report_path,
        )
    else: raise ValueError(f'Unsupported dataset type: {dataset_type}')
    print(desc.to_string(header=False))


if __name__ == '__main__':
    process_corpus_file(
        parquet_path=data_path('tokenizer', 'source_corpus', 'sft_data.parquet'),
        dataset_type='sft',
        filter_rate=0.16,
        len_range=(128, 5120),
        dtype=numpy.uint16,
    )
