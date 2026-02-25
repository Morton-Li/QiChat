"""
scripts/dataset/deduplicate_corpus.py

Tokenization-free chunk-level near-dup removal using MinHash + banded LSH bucketing.
Focus: remove low-entropy/template-heavy dense similarity regions.

Dependencies:
  - lshcurator
  - numpy
  - pandas
  - tqdm
"""
import argparse
import os
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy
import pandas
from lshcurator import Bucket, Deduper
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from scripts.dataset.utils import stream_corpus_batches, split_corpus
from src.utils.path import data_path


def compute_hot_keys_for_corpus_file(
    file_path: Path,
    temp_dir: Path,
    shingle_k: int,
    shingle_step: int,
    bands: int,
    rows_per_band: int,
    compute_mode: Literal['char', 'byte'],
) -> str:
    bucket = Bucket(
        shingle_k=shingle_k,
        shingle_step=shingle_step,
        bands=bands,
        rows_per_band=rows_per_band,
        compute_mode=compute_mode
    )

    for batch in stream_corpus_batches(parquet_path=file_path, batch_size=8000, text_field='text'):
        for text in batch['text']:
            text = normalize_text(text)
            for text_chunk in split_corpus(text=text, max_length=384):
                bucket.insert(text_chunk)

    hot_keys: numpy.ndarray = bucket.extract_keys(min_hit_count=2)  # 只考虑至少出现两次以及以上的桶
    hot_keys_file = f'{file_path.stem}_hot_keys.npy'
    numpy.save(temp_dir / hot_keys_file, hot_keys)

    return hot_keys_file


def normalize_text(text: str) -> str:
    """
    Normalize text for deduplication:
      - Strip leading/trailing whitespace
      - Unicode normalization (NFKC)
    """
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    return text


def main(
    lang: Literal['zh', 'en'],
    subdir: list[str] | None = None,
    max_jobs: int = 1,
):
    if subdir is None: subdir = []
    corpus_files = sorted(list(data_path('tokenizer', 'source_corpus', *subdir).glob(f'**/*.parquet')))
    if not corpus_files:
        print(f'No corpus files found.')
        raise SystemExit(0)
    print(f'Found {len(corpus_files)} corpus files.')

    lang_deduper_kwargs = {
        'zh': {
            'shingle_k': 5,
            'shingle_step': 1,
        },
        'en': {
            'shingle_k': 10,
            'shingle_step': 3,
        },
    }

    # 统计热点桶
    hot_keys_files: list[str] = []
    hot_keys_tmp_path = data_path('tokenizer', 'source_corpus', *subdir, 'hot_keys_tmp')
    if hot_keys_tmp_path.exists():
        for existing_file in hot_keys_tmp_path.glob('**/*'):
            if existing_file.is_file(): existing_file.unlink()
        hot_keys_tmp_path.rmdir()
    hot_keys_tmp_path.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max_jobs) as executor:
        futures = [
            executor.submit(
                compute_hot_keys_for_corpus_file,
                file_path=corpus_file,
                temp_dir=hot_keys_tmp_path,
                **lang_deduper_kwargs[lang],
                bands=16,
                rows_per_band=8,
                compute_mode='char',
            ) for corpus_file in corpus_files
        ]
        for future in tqdm(
            as_completed(futures),
            desc='Computing hot keys for corpus files',
            unit='file',
            leave=False,
            dynamic_ncols=True,
            total=len(futures),
        ):
            hot_keys_files.append(future.result())

    hot_keys: list[numpy.ndarray] = []
    for hot_keys_file in hot_keys_files:
        hot_keys_file_path = hot_keys_tmp_path / hot_keys_file
        hot_keys.append(numpy.load(hot_keys_file_path))
        hot_keys_file_path.unlink(missing_ok=True)
    # 合并所有 corpus 文件的热点桶，得到全局热点桶列表
    global_hot_keys = numpy.unique(numpy.concatenate(hot_keys))
    print(f'Total unique hot keys across all corpus files: {len(global_hot_keys)}')

    deduper = Deduper(
        bucket_keys=global_hot_keys,
        bands=16,
        rows_per_band=8,
        **lang_deduper_kwargs[lang],
        similarity_threshold=0.82,
        compute_mode='char',
        max_representatives_per_bucket=32,
    )

    corpus: list[str] = []
    processed: int = 0
    kept_total: int = 0
    with tqdm(
        total=len(corpus_files),
        desc='Processing corpus files',
        unit='file',
        dynamic_ncols=True,
        # leave=False
    ) as pbar:
        for corpus_file in corpus_files:
            for batch in stream_corpus_batches(parquet_path=corpus_file, batch_size=8000, text_field='text'):
                for text in batch['text']:
                    text = normalize_text(text)
                    # 重复度检测结果列表
                    deduplication_results: list[bool] = [
                        deduper(text=text_chunk) for text_chunk in split_corpus(text=text, max_length=384)
                    ]

                    # 防止极端情况下 split_corpus 产出空列表导致除 0
                    if not deduplication_results:
                        processed += 1
                        continue
                    # 如果被认为是近似重复的文本块占比超过 35%，则丢弃整条文本；否则保留整条文本
                    if deduplication_results.count(False) / len(deduplication_results) < 0.35:
                        corpus.append(text)
                        kept_total += 1
                        processed += 1

                    if processed % 10000 == 0:
                        pbar.write(f'Processed {processed} samples, kept_total: {kept_total}, current deduplication ratio: {1 - kept_total / processed:.2%}')

            pandas.DataFrame({
                'text': corpus
            }).to_parquet(data_path('dataset', f'{corpus_file.stem}_deduplicated.parquet'), index=False)
            corpus.clear()  # 写入文件后清空内存中的 corpus 列表
            pbar.update(n=1)

    print(f'Total unique samples in deduplicated corpus: {len(corpus)}, Total processed samples: {processed}, Deduplication ratio: {1 - len(corpus) / processed:.2%}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deduplicate corpus using MinHash + LSH')
    parser.add_argument('--lang', type=str, default='zh', help='Language code (default: zh)')
    parser.add_argument('--subdir', type=str, nargs='*', default=None, help='Optional subdirectory under tokenizer/source_corpus to look for parquet files')
    parser.add_argument('--max-jobs', type=int, default=max(1, os.cpu_count() - 1), help='Maximum number of processes to run in parallel')
    args = parser.parse_args()

    main(
        lang=args.lang,
        subdir=args.subdir,
        max_jobs=args.max_jobs,
    )
