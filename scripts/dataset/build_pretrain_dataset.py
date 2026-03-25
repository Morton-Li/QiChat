"""
Script to build a dataset from source corpus parquet files.

该脚本会从 data/tokenizer/source_corpus 目录下读取所有 parquet 文件，对文本进行切割和 token 化，并将结果保存为 parquet 文件，位于 data/dataset 目录下。
脚本属于 CPU + Memory 密集型任务，建议在高性能机器上运行。
硬盘使用峰值大约为源数据的两倍，为了避免文件分布不均衡，会将源数据切割成多个相等行数较小 chunk 文件，避免某个子进程处理过多数据导致瓶颈，运行结束后会自动删除这些 chunk 文件。
如果硬盘空间紧张，可在 Preprocessing corpus files 阶段结束后清理源数据，此时不会影响后续的 Tokenizing 阶段。

Usage:
    python scripts/dataset/build_pretrain_dataset.py
        [--text-field TEXT_FIELD]
        [--max-length MAX_LENGTH]
        [--max-jobs MAX_JOBS]
        [--chunk-size CHUNK_SIZE]
        [--no-start-with-bos-token]
        [--dataset-name DATASET_NAME]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy
import pandas
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.tokenizer import QiTianTokenizerFast, get_tokenizer
from src.utils.path import data_path
from scripts.dataset.utils import iter_parquet_batches, auto_select_precision_from_range, generate_report, split_corpus


def process_corpus_file(
    parquet_path: Path|str,
    chunk_size: int,
    chunk_tmp_path: Path,
    text_field: Optional[str|list[str]] = None,
) -> list[Path]:
    """Process a single corpus parquet file into chunks."""
    if isinstance(parquet_path, str): parquet_path = Path(parquet_path)
    if isinstance(text_field, str): text_field = [text_field]

    buffer_list: list[str] = []
    chunk_file_list: list[Path] = []

    for batch in iter_parquet_batches(parquet_path=parquet_path, batch_size=2048, text_field=text_field):
        batch = batch.astype('string').apply(lambda c: c.str.strip())  # 去除前后空白符
        batch = batch.stack()  # 展平为单列（MultiIndex）
        batch = batch.replace('', numpy.nan).dropna()  # 将空字符串替换为 NaN, 去除NaN并重置索引
        buffer_list.extend(batch.tolist())

        if len(buffer_list) >= chunk_size:
            for _ in range(len(buffer_list) // chunk_size):
                chunk_path = chunk_tmp_path / f"{parquet_path.stem}_chunk_{len(chunk_file_list):04d}.parquet"
                # 切割到 chunk_size 大小
                pandas.DataFrame(data={'content': buffer_list[:chunk_size]}).to_parquet(chunk_path)
                chunk_file_list.append(chunk_path)
                buffer_list = buffer_list[chunk_size:]  # 保留剩余部分

    # 处理剩余部分
    if len(buffer_list) > 0:
        chunk_path = chunk_tmp_path / f"{parquet_path.stem}_chunk_{len(chunk_file_list):04d}.parquet"
        pandas.DataFrame(data={'content': buffer_list}).to_parquet(chunk_path)
        chunk_file_list.append(chunk_path)

    return chunk_file_list


def tokenize_chunk_file(
    parquet_path: Path,
    max_length: int,
    start_with_bos_token: bool = True,
) -> tuple[int, int, list[int]]:
    """Tokenize a single chunk parquet file."""
    tokenizer: QiTianTokenizerFast = get_tokenizer()
    if not start_with_bos_token:  # 如果不以 bos token 开头，则需要调整 post-processing 模板，去掉开头的 bos token
        from tokenizers.processors import TemplateProcessing
        tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
            single="$A <|eos|>",
            pair="<|bos|> $A <|eos|> $B <|eos|>",
            special_tokens=[
                (tokenizer.bos_token, tokenizer.bos_token_id),
                (tokenizer.eos_token, tokenizer.eos_token_id),
            ],
        )

    min_token_id, max_token_id = numpy.inf, -numpy.inf
    tokenized_ids: list[list[int]] = []
    sample_lens: list[int] = []
    for text in pandas.read_parquet(parquet_path)['content']:
        text = text.strip()
        if not text: continue
        tokens: list[int] = tokenizer.encode(text)
        if len(tokens) > max_length:  # 如果 token 化后的长度超过 max_length，则进行切割
            for split_text in split_corpus(text=text, max_length=max_length):
                # split_corpus 已做严格的空白符处理，这里不需要再次 strip() 和 if 判断“空”
                split_tokens: list[int] = tokenizer.encode(split_text)
                tokenized_ids.append(split_tokens)
                sample_lens.append(len(split_tokens))
                min_token_id, max_token_id = min(min_token_id, min(split_tokens)), max(max_token_id, max(split_tokens))
        else:  # 否则直接添加进语料列表
            tokenized_ids.append(tokens)
            sample_lens.append(len(tokens))
            min_token_id, max_token_id = min(min_token_id, min(tokens)), max(max_token_id, max(tokens))

    pandas.DataFrame(data={'tokenized': tokenized_ids}).to_parquet(parquet_path, index=False)

    return min_token_id, max_token_id, sample_lens


def build_dataset(
    text_field: Optional[str|list[str]] = None,
    max_length: int = 4096,
    chunk_size: int = 10000,
    max_jobs: int = 1,
    start_with_bos_token: bool = True,
    dataset_name: Optional[str] = None,
    dtype: Optional[numpy.dtype] = None
) -> None:
    """
    Convert the entire corpus into chunks
    Args:
        text_field: The field(s) in the parquet files to load text from. If None, load all fields.
        max_length: Maximum length of tokenized sequences.
        chunk_size: Number of texts per chunk file.
        max_jobs: Maximum number of parallel jobs. Default is 1.
        start_with_bos_token:
            Whether the tokenized sequences start with a BOS token.
            如果为 False，则会调整 tokenizer 的 post-processing 模板，使得生成的 token 序列不以 bos token 开头，这样在后续处理时就不需要再额外去除 bos token。
            这对于某些模型（如 LLaMA）可能更合适，因为它们的训练数据通常不以 bos token 开头。
            如果为 True，则使用默认的 post-processing 模板，生成的 token 序列以 bos token 开头。
        dataset_name: Name of the output dataset. If None, defaults to 'dataset'.
        dtype: Numpy dtype for storing token ids. If None, auto-select based on token id range.
    """
    if not isinstance(text_field, (str, list, type(None))):
        raise TypeError('text_field must be a string or a list of strings or None.')
    print(f'Using up to {max_jobs} parallel jobs for processing.')
    if not start_with_bos_token:
        print('Will adjust tokenizer post-processing to generate sequences without BOS token at the beginning.')
    print(f'Loading {"all fields" if text_field is None else f"field(s) ({', '.join(text_field)})"} from all source corpus parquet files...')
    # 检查目录是否为空
    file_list = list(data_path('tokenizer', 'source_corpus').glob('**/*.parquet'))
    if not file_list: raise FileNotFoundError(f'No parquet files found in source corpus path.')
    file_list.sort()

    dataset_name = dataset_name or 'dataset'

    chunk_tmp_path = data_path('tokenizer', 'source_corpus', 'chunks')
    chunk_tmp_path.mkdir(parents=True, exist_ok=True)
    chunk_file_list: list[Path] = []
    with ProcessPoolExecutor(max_workers=max_jobs) as executor:
        futures = [
            executor.submit(
                process_corpus_file,
                parquet_path=parquet_path, chunk_size=chunk_size, chunk_tmp_path=chunk_tmp_path, text_field=text_field,
            ) for parquet_path in file_list
        ]
        for future in tqdm(
            as_completed(futures),
            desc='Preprocessing corpus files',
            leave=True,
            dynamic_ncols=True,
            total=len(futures),
        ):
            chunk_file_list.extend(future.result())

    # 切割并token化
    min_token_id, max_token_id = numpy.inf, -numpy.inf
    sample_lens: list[int] = []
    with ProcessPoolExecutor(max_workers=max_jobs) as executor:
        futures = [
            executor.submit(
                tokenize_chunk_file,
                parquet_path=parquet_path, max_length=max_length, start_with_bos_token=start_with_bos_token,
            ) for parquet_path in chunk_file_list
        ]
        for future in tqdm(
            as_completed(futures),
            desc=f'Processing corpus',
            leave=True,
            dynamic_ncols=True,
            total=len(futures),
        ):
            future_min_token_id, future_max_token_id, future_sample_lens = future.result()
            min_token_id = min(min_token_id, future_min_token_id)
            max_token_id = max(max_token_id, future_max_token_id)
            sample_lens.extend(future_sample_lens)

    # 自动选择合适的 dtype
    if dtype is None:
        dtype = auto_select_precision_from_range(min_token_id, max_token_id)
    print(f'Chosen dtype for token storage: {dtype.__name__}  (range: {min_token_id} ~ {max_token_id})')

    tokenized = []
    for parquet_path in chunk_file_list:
        for seq in pandas.read_parquet(parquet_path)['tokenized'].tolist():
            tokenized.append(numpy.array(seq, dtype=dtype))
        parquet_path.unlink(missing_ok=True)  # 处理完后删除 chunk 文件以节省空间

    pandas.DataFrame(data={'tokenized': tokenized}).to_parquet(data_path('dataset', f'{dataset_name}.parquet'))
    print(f'Dataset built successfully.')

    # 删除临时目录
    chunk_tmp_path.rmdir()

    # 简报
    desc = generate_report(
        sample_lens=sample_lens,
        save_path=data_path('dataset', f'{dataset_name}.json')
    )
    print(desc.to_string(header=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='构建数据集脚本，该脚本会从 data/tokenizer/source_corpus 目录下读取所有 parquet 文件并进行处理。')

    parser.add_argument(
        "--text-field",
        type=str,
        default=None,
        help="设置要从 parquet 文件中加载文本的字段名称（默认：加载所有字段）。",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="设置最大序列长度的整数值（默认：4096）。",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=max(1, os.cpu_count() - 1),
        help="设置并行处理进程数量（默认：CPU 核心数减1）。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="设置每个切块文件中的行数（默认：10000）。",
    )
    parser.add_argument(
        "--no-start-with-bos-token",
        dest="start_with_bos_token",
        action="store_false",
        help="设置生成的 token 序列是否*不*以 BOS token 开头。如果传递该参数，则会调整 tokenizer 的 post-processing 模板，使得生成的 token 序列不以 BOS token 开头。",
    )
    parser.set_defaults(start_with_bos_token=True)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="设置输出数据集的名称（默认：dataset）。",
    )
    args = parser.parse_args()

    build_dataset(
        text_field=args.text_field.split(',') if args.text_field else None,
        max_length=args.max_length,
        chunk_size=args.chunk_size,
        max_jobs=args.max_jobs,
        start_with_bos_token=args.start_with_bos_token,
        dataset_name=args.dataset_name,
        dtype=None
    )
