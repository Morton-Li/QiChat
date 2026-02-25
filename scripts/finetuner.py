#!/usr/bin/env python3
"""
Script to train the QiChat model.

Usage:
    python scripts/finetuner.py [--nprocs N]
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.trainer import FineTuner, ddp_worker

if __name__ == '__main__':
    print('\033[0;32m ------------------------------\033[0m')
    print('\033[0;32m|          >>> Qi <<<          |\033[0m')
    print('\033[0;32m ------------------------------\033[0m')

    parser = argparse.ArgumentParser(description='微调 QiChat 模型')
    parser.add_argument('--nprocs', type=int, default=1, help='使用nproc个GPU进行训练')
    args = parser.parse_args()

    try:
        nprocs = args.nprocs
        if nprocs < 1: raise ValueError('nprocs must be at least 1.')
        elif nprocs == 1:
            trainer = FineTuner()
            trainer.start()
        elif nprocs > 1:
            import torch
            from torch import multiprocessing
            world_size = torch.cuda.device_count()
            if nprocs > world_size:
                raise ValueError(f'nprocs ({nprocs}) cannot be greater than available GPUs ({world_size}).')
            multiprocessing.spawn(
                fn=ddp_worker,
                args=(nprocs, FineTuner),
                nprocs=nprocs,
                join=True
            )
    except KeyboardInterrupt:
        print()
        print('\033[0;31m训练被打断。\033[0m')
