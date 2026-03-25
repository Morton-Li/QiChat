#!/usr/bin/env python3
"""
加载 `2.select_bucket_keys.py` 产物，并基于筛出的 deduper bucket keys 与逐文件 should-dedupe 掩码，重新扫描现有语料文件，执行真正的 deduplication.

Dependencies:
  - lshcurator >= 0.2.2
  - numpy
  - tqdm
  - pandas
  - pyarrow
"""
import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy
import pandas
from tqdm import tqdm

try:
    from lshcurator import Deduper, DeduperConfig
    from lshcurator.utils.readers import iter_corpus_texts
except ImportError:
    print('lshcurator not installed. Please install it via `pip install lshcurator` and try again.')
    sys.exit(1)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.utils.path import data_path
from scripts.dataset.dedup.utils import format_count, rel_to_data_dir

LANG_DEDUPER_KWARGS: dict[Literal['zh', 'en'], dict[str, int]] = {
    'zh': {'shingle_k': 5, 'shingle_step': 1},
    'en': {'shingle_k': 10, 'shingle_step': 3},
}


def normalize_mask_mapping(mapping: numpy.ndarray) -> dict[Path, numpy.typing.NDArray[numpy.bool_]]:
    if isinstance(mapping, numpy.ndarray) and mapping.dtype == object:
        if mapping.shape == (): mapping = mapping.item()
        elif mapping.size == 1: mapping = mapping.reshape(()).item()
        else: raise ValueError(f'Expected file_should_dedupe_masks to be a 0D or 1D object array containing a dict, but got shape {mapping.shape} with dtype {mapping.dtype}.')

    if not isinstance(mapping, dict):
        raise TypeError(f'Expected file_should_dedupe_masks payload to be dict, but got {type(mapping).__name__}.')

    normalized: dict[Path, numpy.typing.NDArray[numpy.bool_]] = {}
    for raw_path, raw_mask in mapping.items():
        file_path = data_path(raw_path)
        mask = numpy.asarray(raw_mask, dtype=numpy.bool)
        if mask.ndim != 1: raise ValueError(f'Expected file mask for {file_path} to be 1D, but got shape {mask.shape}.')
        normalized[file_path] = mask
    return normalized


def user_confirmation(prompt: str) -> bool:
    while True:
        user_input = input(prompt + ' (y/n): ').strip().lower()
        if user_input == 'y': return True
        elif user_input == 'n': return False
        else: print('Invalid input. Please enter "y" for yes or "n" for no.')


def main(
    lang: Literal['zh', 'en'],
    similarity_threshold: float,
    subdir: list[str] | None = None,
    text_field: list[str] | None = None,
    output_dir: Path| str | None = None,
    bands: int = 16,
    rows_per_band: int = 8,
    max_representatives_per_bucket: int | None = None,
) -> None:
    # 0. 前置检查
    if not (0 <= similarity_threshold <= 1): raise ValueError(f'similarity_threshold must be in [0, 1], but got {similarity_threshold}.')
    if max_representatives_per_bucket is not None and max_representatives_per_bucket < 1: raise ValueError('max_representatives_per_bucket must be at least 1 when provided.')

    deduper_bucket_keys_path = data_path('dataset', 'dedup_cache', 'select_bucket_keys', 'deduper_bucket_keys.npz')
    if not deduper_bucket_keys_path.exists(): raise FileNotFoundError(f'deduper_bucket_keys artifact not found: {rel_to_data_dir(deduper_bucket_keys_path)}')
    file_should_dedupe_masks_path = data_path('dataset', 'dedup_cache', 'select_bucket_keys', 'file_should_dedupe_masks.npz')
    if not file_should_dedupe_masks_path.exists(): raise FileNotFoundError(f'file_should_dedupe_masks artifact not found: {rel_to_data_dir(file_should_dedupe_masks_path)}')

    if subdir is None: subdir = []
    corpus_dir = data_path('tokenizer', 'source_corpus', *subdir)
    if output_dir is not None:
        if isinstance(output_dir, str): output_dir = Path(output_dir)
        if not isinstance(output_dir, Path): raise ValueError(f'Expected output_dir to be a Path or str, but got {type(output_dir).__name__}.')
    else: output_dir = data_path('dataset')

    # 1. 加载 deduper_bucket_keys 和 file_should_dedupe_masks
    deduper_bucket_keys: numpy.typing.NDArray[numpy.uint64] = numpy.load(file=deduper_bucket_keys_path)['deduper_bucket_keys']  # (deduper_bucket_keys,) 1D array of uint64
    if deduper_bucket_keys.ndim != 1: raise ValueError(f'Expected deduper_bucket_keys to be 1D, but got shape {deduper_bucket_keys.shape}.')
    if deduper_bucket_keys.size == 0:  # 在此脚本场景下不应允许没有 deduper_bucket_keys 的情况，因为这代表整个样本没有一行是重复的
        print(f'Loaded deduper_bucket_keys is empty. This means no rows are selected as deduplication candidates, and all rows will be treated as unique. If this is unexpected, please check the previous step `2.select_bucket_keys.py` to ensure it has correctly identified deduplication candidates based on the computed bucket keys.')
        raise SystemExit(0)
    print(f'Loaded {deduper_bucket_keys.size} deduper bucket keys.')

    file_should_dedupe_masks: dict[str, numpy.typing.NDArray[numpy.bool_]] = numpy.load(
        file_should_dedupe_masks_path, allow_pickle=True
    )['file_should_dedupe_masks'].item()  # key 为相对 data_path() 的 str 路径，可直接使用 data_path 函数转 Path
    print(f'Loaded should-dedupe masks for {len(file_should_dedupe_masks)} files.')

    # 2. 初始化 Deduper
    deduper = Deduper(
        bucket_keys=deduper_bucket_keys,
        config=DeduperConfig(
            **LANG_DEDUPER_KWARGS[lang],
            bands=bands,
            rows_per_band=rows_per_band,
            similarity_threshold=similarity_threshold,
            compute_mode='char',
            max_representatives_per_bucket=max_representatives_per_bucket,
        ),
    )

    snapshot_path = output_dir / 'deduper_buckets.npz'
    if snapshot_path.exists():
        print(f'Found existing deduper buckets snapshot at {snapshot_path}. Loading buckets to resume deduplication process...')
        snapshot = numpy.load(snapshot_path, allow_pickle=True)
        deduper._buckets = snapshot['buckets'].item()
        total_rows = int(snapshot['total_rows'])
        total_unique_rows = int(snapshot['total_unique_rows'])
        total_duplicated_rows = int(snapshot['total_duplicated_rows'])
        total_processed_rows = int(snapshot['total_processed_rows'])
        processed_files = set(snapshot['processed_files'])
        snapshot.close()
        print(f'Successfully loaded deduper buckets from snapshot. Resuming deduplication process with {deduper.num_buckets} buckets already populated.')
    else:
        print('No existing deduper buckets snapshot found. Starting deduplication process from scratch...')
        total_rows = 0
        total_unique_rows = 0
        total_duplicated_rows = 0
        total_processed_rows = 0
        # 已处理文件名列表
        processed_files: set[str] = set()

    # 3. 扫描语料目录下的 corpus 文件，执行 deduplication
    while True:
        batch_corpus_files: list[Path] = [file for file in sorted(corpus_dir.glob('**/*.parquet')) if file.name not in processed_files]
        if len(batch_corpus_files) == 0:
            # 在语料目录下没有找到任何新的 corpus 文件，询问用户确认是否检查目录内容并重试，或者直接结束
            if user_confirmation(
                f'No new corpus files found in {rel_to_data_dir(corpus_dir)}. '
                f'Do you want to retry checking for corpus files after confirming the directory contents?'
            ): continue
            else:
                if total_rows > 0: numpy.savez_compressed(
                    output_dir / 'deduper_buckets',
                    buckets=deduper.buckets,
                    total_rows=total_rows, total_unique_rows=total_unique_rows, total_duplicated_rows=total_duplicated_rows, total_processed_rows=total_processed_rows,
                    processed_files=processed_files,
                )
                break
        print(f'Found {len(batch_corpus_files)} corpus files.')

        with tqdm(total=len(batch_corpus_files), desc='Processing corpus files', unit='file', dynamic_ncols=True) as pbar:
            for file_path in batch_corpus_files:
                file_mask = file_should_dedupe_masks.get(str(rel_to_data_dir(file_path)), None)
                if file_mask is None:
                    # 警告：该文件没有对应的 should-dedupe mask，意味着该文件的所有行都会执行去重逻辑
                    pbar.write(f'Warning: No should-dedupe mask found for file {rel_to_data_dir(file_path)}. All rows in this file will be processed for deduplication, which may take significantly more time. If this is unexpected, please check the output of the previous step `2.select_bucket_keys.py` to ensure it has generated should-dedupe masks for all corpus files.')

                file_total_rows = 0
                file_unique_rows = 0
                file_duplicated_rows = 0
                file_processed_rows = 0

                texts: list[str] = []

                for text in iter_corpus_texts(
                    files_path=file_path,
                    fields=text_field,
                ):
                    if file_mask is None or bool(file_mask[file_total_rows]):  # 是否需要去重
                        is_unique = deduper(str(text))
                        file_processed_rows += 1
                    else: is_unique = True  # 不需要去重的行直接视为唯一

                    if is_unique:
                        texts.append(str(text))
                        file_unique_rows += 1
                    else: file_duplicated_rows += 1

                    file_total_rows += 1

                if file_total_rows != file_mask.size:
                    raise ValueError(
                        f'Reader output is out of sync with file_should_dedupe_masks for file {file_path}. '
                        f'consumed_rows={file_total_rows}, expected_rows={file_mask.size}'
                    )

                pandas.DataFrame({
                    'text': texts,
                }).to_parquet(output_dir / f'{file_path.stem}.parquet', index=False)
                texts.clear()

                total_rows += file_total_rows
                total_unique_rows += file_unique_rows
                total_duplicated_rows += file_duplicated_rows
                total_processed_rows += file_processed_rows
                processed_files.add(file_path.name)

                pbar.set_postfix({
                    'rows': format_count(total_rows),
                    'duplicated': format_count(total_duplicated_rows),
                    'processed': format_count(total_processed_rows),
                    'dup_rate': f'{total_duplicated_rows / total_rows:.2%}' if total_rows > 0 else 'N/A',
                })
                pbar.update(1)

        # 用户增加了语料，直接继续无需确认
        if len([file for file in corpus_dir.glob('**/*.parquet') if file.name not in processed_files]) > 0: continue

        # 询问用户更换语料继续去重还是完成并结束
        if user_confirmation(
            f'Finished processing current batch of corpus files. Do you want to check for more corpus files to process, or are you done and want to finish the deduplication process?'
        ): continue
        else:
            if total_rows > 0: numpy.savez_compressed(
                output_dir / 'deduper_buckets',
                buckets=deduper.buckets,
                total_rows=total_rows, total_unique_rows=total_unique_rows, total_duplicated_rows=total_duplicated_rows, total_processed_rows=total_processed_rows,
                processed_files=processed_files,
            )
            break

    print(f'Unique rows: {format_count(total_unique_rows)}')
    print(f'Duplicated rows: {format_count(total_duplicated_rows)}')
    print(f'Duplication rate: {total_duplicated_rows / total_rows:.2%}' if total_rows > 0 else 'Duplication rate: N/A')
    print(f'Processed rows: {format_count(total_processed_rows)} (rows that were actually checked for duplication among the candidates)')
    print(f'Total rows: {format_count(total_rows)}')
    print(f'Finished processing corpus. Output saved to: {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='执行 deduplication 第二阶段：加载 select_bucket_keys 产物并扫描现有语料文件。')
    parser.add_argument('--lang', type=str, choices=['zh', 'en'], default='zh', help='语料语言，决定 DeduperConfig 中 shingle_k 和 shingle_step 的默认值；默认 zh。')
    parser.add_argument('--subdir', type=str, nargs='*', default=None, help='语料子目录列表，用于限定处理范围；默认处理所有子目录下的语料文件。')
    parser.add_argument('--similarity-threshold', type=float, required=True, help='Jaccard/MinHash 相似度阈值，范围 [0, 1]。')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录；默认 data/dataset。')
    parser.add_argument('--text-field', type=str, nargs='*', default=None, help='显式指定文本字段；默认为全部字段')
    parser.add_argument('--bands', type=int, default=16, help='显式指定 LSH bands 数量，默认为 16。')
    parser.add_argument('--rows-per-band', type=int, default=8, help='显式指定 LSH rows_per_band，默认为 8。')
    parser.add_argument('--max-representatives-per-bucket', type=int, default=None, help='每个 bucket 最多保留多少代表样本。默认不限制。')
    args = parser.parse_args()

    main(
        lang='zh',
        similarity_threshold=args.similarity_threshold,
        subdir=args.subdir,
        output_dir=args.output_dir,
        text_field=args.text_field,
        bands=args.bands,
        rows_per_band=args.rows_per_band,
        max_representatives_per_bucket=args.max_representatives_per_bucket,
    )
