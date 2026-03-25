#!/usr/bin/env python3
"""
该脚本使用 lshcurator 分批次对语料计算 bucket-key 供后续 deduplication 使用。

Dependencies:
  - lshcurator >= 0.2.2
  - pandas
  - tqdm
"""
import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy
try:
    from lshcurator import Curator, CuratorConfig
except ImportError:
    print('lshcurator not installed. Please install it via `pip install lshcurator` and try again.')
    sys.exit(1)
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.utils.path import data_path
from scripts.dataset.dedup.utils import format_count, rel_to_data_dir


LANG_DEDUPER_KWARGS: dict[Literal['zh', 'en'], dict[str, int]] = {
    'zh': {'shingle_k': 5, 'shingle_step': 1},
    'en': {'shingle_k': 10, 'shingle_step': 3},
}


def main(
    lang: Literal['zh', 'en'],
    subdir: list[str] | None = None,
    text_field: list[str] | None = None,
    max_jobs: int = 1,
    bands: int = 16,
    rows_per_band: int = 8,
    **kwargs: dict[str, Any],
) -> None:
    # 0. Validate arguments and warn about potential resource issues
    if max_jobs < 1: raise ValueError('max_jobs must be at least 1.')
    files_per_batch = kwargs.get('files_per_batch', None)
    if files_per_batch is None: files_per_batch = max_jobs
    if files_per_batch < 1: raise ValueError('files_per_batch must be at least 1.')
    if max_jobs > os.cpu_count(): print(f'Warning: max_jobs ({max_jobs}) is greater than available CPU cores ({os.cpu_count()}). This may lead to resource contention and slowdowns.')
    if max_jobs > files_per_batch: print(f'Warning: max_jobs ({max_jobs}) is greater than files_per_batch ({files_per_batch}). This may lead to underutilization of worker processes.')

    # 1. Discover corpus files
    if subdir is None: subdir = []
    corpus_file_dir = data_path('tokenizer', 'source_corpus', *subdir)
    corpus_files = sorted(corpus_file_dir.glob('**/*.parquet'))
    if not corpus_files:
        print('No corpus files found.')
        raise SystemExit(0)
    print(f'Found {len(corpus_files)} corpus files.')

    # 2. Prepare output directory and manifest
    output_dir = data_path('dataset', 'dedup_cache', 'compute_bucket_keys')
    manifest_path = output_dir / 'manifest.json'
    if manifest_path.exists(): raise FileExistsError(f'Bucket index manifest already exists: {manifest_path}')
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_path = output_dir / 'batches'
    batch_path.mkdir(parents=True, exist_ok=True)

    file_batches = [corpus_files[i: i + files_per_batch] for i in range(0, len(corpus_files), files_per_batch)]
    manifest: dict[str, Any] = {
        'lang': lang,
        'text_field': text_field if text_field else 'ALL',
        'config': {
            **LANG_DEDUPER_KWARGS[lang],
            'bands': bands, 'rows_per_band': rows_per_band,
        },
        'file_batches': [
            [str(rel_to_data_dir(file)) for file in files]
            for files in file_batches
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Writing bucket-key artifacts into: {output_dir}')

    # 3. Batch compute bucket keys and file-bucket mappings, and persist them
    curator = Curator(
        config=CuratorConfig(
            **LANG_DEDUPER_KWARGS[lang],
            bands=bands,
            rows_per_band=rows_per_band,
            compute_mode='char',
            max_workers=max_jobs,
        )
    )

    total_rows = 0
    with tqdm(total=len(file_batches), desc='Computing bucket-key batches', unit='batch', dynamic_ncols=True) as pbar:
        for batch_index, batch_files in enumerate(file_batches):
            bucket_keys, file_bucket_pos_mapping = curator.compute_bucket_keys(
                files_path=batch_files,
                fields=text_field,
                key_layout='row_bands',
            )
            total_rows += bucket_keys.shape[0]

            numpy.savez_compressed(batch_path / f'batch_{batch_index:05d}_bucket_keys', bucket_keys=bucket_keys)
            mapping_payload = {
                str(rel_to_data_dir(file_path)): [asdict(chunk) for chunk in chunks]
                for file_path, chunks in file_bucket_pos_mapping.items()
            }
            (batch_path / f'batch_{batch_index:05d}_file_bucket_pos_mapping.json').write_text(
                json.dumps(mapping_payload, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )

            pbar.set_postfix({'total_rows': format_count(total_rows)})
            pbar.update(1)

    print(f'Total rows covered by saved bucket keys: {format_count(total_rows)}')
    print(f'Finished. Manifest saved to: {output_dir / "manifest.json"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch-compute and persist bucket-key artifacts for deduplication.')
    parser.add_argument('--lang', type=str, default='zh', choices=['zh', 'en'], help='Language code.')
    parser.add_argument('--text-field', type=str, nargs='*', default=None, help='Text field(s) in parquet files. Default: all text fields.')
    parser.add_argument('--subdir', type=str, nargs='*', default=None, help='Optional subdirectory under tokenizer/source_corpus to scan.')
    parser.add_argument('--max-jobs', type=int, default=max(1, (os.cpu_count() or 1) - 1), help='Maximum number of worker processes used by curator.')
    parser.add_argument('--bands', type=int, default=16, help='Number of bands for LSH. Higher means more precision but less recall.')
    parser.add_argument('--rows-per-band', type=int, default=8, help='Number of rows per band for LSH. Higher means more precision but less recall.')
    parser.add_argument('--files-per-batch', type=int, default=None, help='How many parquet files to pass into one compute_bucket_keys call.')
    args = parser.parse_args()

    main(
        lang=args.lang,
        text_field=args.text_field,
        subdir=args.subdir,
        max_jobs=args.max_jobs,
        bands=args.bands,
        rows_per_band=args.rows_per_band,
        files_per_batch=args.files_per_batch,
    )
