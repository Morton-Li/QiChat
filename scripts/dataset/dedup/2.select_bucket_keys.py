#!/usr/bin/env python3
"""
聚合 `compute_bucket_keys.py` 的全部批次产物，使用 `Curator._select_deduper_bucket_keys`
过滤低频 bucket keys，并保存第二阶段 deduplication 所需的 key 集合与逐批次行掩码。

Dependencies:
  - lshcurator >= 0.2.2
  - numpy
  - tqdm
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy

try:
    from lshcurator import Curator
except ImportError:
    print('lshcurator not installed. Please install it via `pip install lshcurator` and try again.')
    sys.exit(1)
from lshcurator.utils.types import BucketKeyChunk
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from src.utils.path import data_path
from scripts.dataset.dedup.utils import format_count, rel_to_data_dir


def load_bucket_keys(path: Path) -> numpy.typing.NDArray[numpy.uint64]:
    payload = numpy.load(path, allow_pickle=False)
    if isinstance(payload, numpy.ndarray):
        bucket_keys = payload
    else:
        try:
            if 'bucket_keys' in payload.files: bucket_keys = payload['bucket_keys']
            else: raise KeyError(f'Expected `bucket_keys` in {path}, but found keys: {payload.files}.')
        finally:
            payload.close()

    bucket_keys = numpy.asarray(bucket_keys, dtype=numpy.uint64)
    if bucket_keys.ndim != 2:
        raise ValueError(f'Expected bucket_keys in {path} to be a 2D array with shape (num_samples, bands), but got shape {bucket_keys.shape}.')
    return bucket_keys


def main(
    filter_freq: int = 1,
) -> None:
    if filter_freq < 0: raise ValueError(f'filter_freq must be a non-negative integer, but got {filter_freq}.')

    compute_result_path = data_path('dataset', 'dedup_cache')
    compute_result_dirs = [path for path in compute_result_path.glob('**/compute_bucket_keys') if path.is_dir()]
    if len(compute_result_dirs) == 0:
        print(f'No compute_bucket_keys result directories found in {compute_result_path}.')
        raise SystemExit(0)
    print(f'Found {len(compute_result_dirs)} compute_bucket_keys directories')
    output_dir = data_path('dataset', 'dedup_cache', 'select_bucket_keys')
    if output_dir.exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}. Please remove it before running this script to avoid accidentally overwriting existing results.')
    output_dir.mkdir(parents=True, exist_ok=True)

    all_bucket_keys: list[numpy.ndarray] = []
    batch_stats: list[dict[str, Any]] = []
    total_rows = 0
    expected_bands = 16

    with tqdm(total=len(compute_result_dirs), desc='Processing compute_bucket_keys results', unit='result_dir', dynamic_ncols=True) as pbar:
        for result_dir in compute_result_dirs:
            source_manifest_path = result_dir / 'manifest.json'
            if not source_manifest_path.exists():
                raise FileNotFoundError(f'compute_bucket_keys manifest not found: {source_manifest_path}')
            source_manifest: dict[str, Any] = json.loads(source_manifest_path.read_text(encoding='utf-8'))

            file_batches = source_manifest.get('file_batches', None)
            if not isinstance(file_batches, list):
                raise ValueError(f'Invalid source manifest: `file_batches` must be a list, but got {type(file_batches).__name__}.')

            batch_dir = result_dir / 'batches'
            if not batch_dir.exists():
                raise FileNotFoundError(f'Batch artifact directory not found: {batch_dir}')

            with tqdm(total=len(file_batches), desc='Loading bucket-key batches', unit='batch', dynamic_ncols=True, leave=False) as batch_pbar:
                for batch_index, batch_files in enumerate(file_batches):
                    artifact_path = batch_dir / f'batch_{batch_index:05d}_bucket_keys.npz'
                    bucket_keys = load_bucket_keys(artifact_path)
                    file_bucket_pos_mapping = json.loads((batch_dir / f'batch_{batch_index:05d}_file_bucket_pos_mapping.json').read_text(encoding='utf-8'))

                    if bucket_keys.shape[1] != expected_bands:
                        raise ValueError(f'Batch {batch_index} has {bucket_keys.shape[1]} bands, but source manifest declares {expected_bands}.')

                    all_bucket_keys.append(bucket_keys)

                    batch_stats.append({
                        'batch_index': batch_index,
                        'source_files': batch_files,
                        'bucket_keys_artifact': rel_to_data_dir(artifact_path),
                        'offset': total_rows,
                        'rows': int(bucket_keys.shape[0]),
                        'bands': int(bucket_keys.shape[1]),
                        'file_bucket_pos_mapping_payload': file_bucket_pos_mapping
                    })

                    total_rows += int(bucket_keys.shape[0])

                    pbar.set_postfix({'total_rows': format_count(total_rows)})
                    batch_pbar.update(1)
            pbar.update(1)

    if len(all_bucket_keys) == 0:
        print('No bucket keys loaded from any batch. Exiting.')
        raise SystemExit(0)
    bucket_keys = numpy.concatenate(all_bucket_keys, axis=0)
    print(f'Loaded {len(all_bucket_keys)} batch(es), total rows: {format_count(int(bucket_keys.shape[0]))}.')

    print(f'Selecting deduper bucket keys with filter_freq={filter_freq}...')
    deduper_bucket_keys, should_dedupe_row_mask = Curator.select_deduper_bucket_keys(bucket_keys=bucket_keys, filter_freq=filter_freq)
    numpy.savez_compressed(output_dir / 'deduper_bucket_keys', deduper_bucket_keys=deduper_bucket_keys)

    print(f'Constructing file-level deduplication masks for {len(batch_stats)} batches...')
    file_should_dedupe_masks: dict[str, numpy.ndarray] = {}
    for batch in batch_stats:
        batch_start_offset: int = batch['offset']
        batch_end: int = batch_start_offset + batch['rows']
        batch_mask = should_dedupe_row_mask[batch_start_offset:batch_end]
        for rel_file_str, chunks in batch['file_bucket_pos_mapping_payload'].items():
            chunks = [BucketKeyChunk(**chunk) for chunk in chunks]
            if not chunks:
                print(f'[WARNING] Skipping file {rel_file_str} with no bucket key chunks.')
                file_should_dedupe_masks[rel_file_str] = numpy.empty(0, dtype=numpy.bool_)
                continue
            file_should_dedupe_mask: list[numpy.ndarray] = []
            file_total_rows: int = 0
            for chunk in chunks:
                file_mask = batch_mask[chunk.start_position: chunk.start_position + chunk.size]
                file_should_dedupe_mask.append(file_mask)
                file_total_rows += chunk.size

            file_should_dedupe_mask: numpy.ndarray = numpy.concatenate(file_should_dedupe_mask, axis=0)
            if file_should_dedupe_mask.shape[0] != file_total_rows:
                raise RuntimeError(
                    f'Internal error when slicing row masks for file {rel_file_str}: expected total rows {file_total_rows} from chunks, '
                    f'but got mask with {file_should_dedupe_mask.shape[0]} rows.'
                )
            if rel_file_str in file_should_dedupe_masks: file_should_dedupe_masks[rel_file_str] = numpy.concatenate([file_should_dedupe_masks[rel_file_str], file_should_dedupe_mask], axis=0)
            else: file_should_dedupe_masks[rel_file_str] = file_should_dedupe_mask
    numpy.savez_compressed(output_dir / 'file_should_dedupe_masks', file_should_dedupe_masks=file_should_dedupe_masks)

    print(f'Deduper bucket keys: {format_count(int(deduper_bucket_keys.size))}')
    print(f'Rows requiring deduplication: {format_count(int(should_dedupe_row_mask.sum()))} / {format_count(int(bucket_keys.shape[0]))}')
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='聚合 compute_bucket_keys 产物，并筛出用于第二阶段 deduplication 的 bucket keys。')
    parser.add_argument('--filter-freq', type=int, default=1, help='过滤掉出现次数 <= 该阈值的 bucket keys。默认: 1')
    args = parser.parse_args()

    main(
        filter_freq=args.filter_freq,
    )
