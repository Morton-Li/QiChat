#!/usr/bin/env python3
"""
Check for duplicate samples in a dataset parquet file based on n-gram duplication ratio.

Usage:
    python scripts/dataset/checks/check_duplicates.py --dataset_name DATASET_NAME [--field_name FIELD_NAME] [--ngram_n N] [--duplicate_threshold THRESHOLD]
"""
import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.dataset.utils import stream_corpus_batches
from src.tokenizer import QiTianTokenizerFast, get_tokenizer
from src.utils.masks import build_span_mask
from src.utils.path import data_path


def dup_ngram_ratio(token_ids: Sequence[int], n: int) -> float:
    """
    Duplicate n-gram ratio = 1 - (#unique ngrams / #total ngrams)
    """
    if n <= 0: return 0.0
    total = len(token_ids) - n + 1
    if total <= 1: return 0.0
    seen = set()
    # tuples are fine for n in [4..6] and seq length ~1k
    for i in range(total): seen.add(tuple(token_ids[i : i + n]))
    return 1.0 - (len(seen) / total)


def main(
    dataset_name: str,
    field_name: str = 'tokenized',
    ngram_n: int = 4,
    duplicate_threshold: float = 0.12,
):
    """
    Check for duplicate samples in a dataset parquet file.
    """
    if not dataset_name.endswith('.parquet'): dataset_name = dataset_name + '.parquet'
    batch_size = 2048

    tokenizer: QiTianTokenizerFast = get_tokenizer()

    duplicate_list = list()
    total_count = 0

    for b_i, batch in enumerate(stream_corpus_batches(
        parquet_path=data_path('dataset', dataset_name),
        batch_size=batch_size,
        text_field=field_name
    )):
        for s_i, token_ids_ndarray in batch.itertuples(index=True):
            total_count += 1

            # True = ast部分, False = 其他角色部分（ignore）
            span_mask = build_span_mask(
                input_ids=token_ids_ndarray,
                start_token_id=tokenizer.convert_tokens_to_ids('<|assistant|>'),
                end_token_id=tokenizer.convert_tokens_to_ids('<|eot|>'),
                include_start_token=False,
                include_end_token=True,
            )
            token_ids = token_ids_ndarray[span_mask].tolist()

            dup_ratio = dup_ngram_ratio(token_ids, ngram_n)
            if dup_ratio > duplicate_threshold:
                duplicate_list.append({
                    'text': tokenizer.decode(token_ids, skip_special_tokens=False).strip(),
                    'token_ids': token_ids,
                    'file_name': dataset_name,
                    'index': total_count - 1,
                    'type': f'ngram_{ngram_n}',
                    'dup_ngram_ratio': dup_ratio,
                })
    duplicate_df = pandas.DataFrame(duplicate_list)
    duplicate_df.to_parquet(
        data_path('dataset', f'duplicates_{dataset_name.replace(".parquet", "")}.parquet'),
        index=False,
    )
    duplicate_count = len(duplicate_df)
    print(f'Total samples processed: {total_count}')
    print(f'Duplicate samples found: {duplicate_count} ({(duplicate_count / total_count):.2%})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check for duplicate samples in a dataset parquet file.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset parquet file.')
    parser.add_argument('--field_name', type=str, default='tokenized', help='Field name containing tokenized data.')
    parser.add_argument('--ngram_n', type=int, default=4, help='N value for n-gram duplication check.')
    parser.add_argument('--duplicate_threshold', type=float, default=0.12, help='Threshold for duplicate n-gram ratio.')
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        field_name=args.field_name,
        ngram_n=args.ngram_n,
        duplicate_threshold=args.duplicate_threshold,
    )
