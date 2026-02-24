#!/usr/bin/env python3
"""
Script to train the Qwen3 model.

Usage:
    python scripts/pretrain.py [--nprocs N]
"""
import argparse
import math
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.trainer import PreTrainer, ddp_worker
from src.probes import ModelProbe


def collect_layer_grad_norms(model: torch.nn.Module):
    """
    Collect L2 norms of gradients for each layer in the model.
    Returns a dict mapping layer keys (e.g., "block_0", "emb", "final_ln") to their corresponding gradient norms.
    模型在配置了 TP / PP  时无效
    """
    # layer_sq[key] = sum of grad^2 for that layer
    layer_sq: dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if param.grad is None: continue

        # 解析层
        if name.startswith("model.layers."):
            parts = name.split(".")
            idx = int(parts[2])  # model.blocks.<idx>.xxx
            key = f"layers_{idx:02d}"
        elif name.startswith("model.embed_tokens."):
            key = "emb"
        elif name.startswith("model.lm_head."):
            key = "final_ln"
        else:
            key = "other"

        # 懒初始化累计器（与梯度在同 device）
        if key not in layer_sq:
            layer_sq[key] = torch.zeros((), device=param.grad.device, dtype=torch.float32)
        # 累加 ||grad||^2（用 float32 统计更稳）
        layer_sq[key] += param.grad.detach().float().pow(2).sum()

    # sqrt(sum) -> L2 norm
    layer_norm = {k: math.sqrt(v.item()) for k, v in layer_sq.items()}
    return layer_norm


# 劫持原引入
PreTrainer.collect_layer_grad_norms = collect_layer_grad_norms


class Qwen3PreTrainer(PreTrainer):
    """ Pre-trainer for Qwen3 Model """

    def init_model(self):
        """Initialize the Qwen3 Model"""
        self.logger.info('Initializing Qwen3 Model ... ', newline=False)
        model_config_kwargs = {
            'attn_implementation': self.config.model.attn_implementation,
            'dtype': self.config.model.dtype,
        }
        model_config_kwargs.update(self.config.model.model_config_kwargs)
        model_config = Qwen3Config(  # 73.27 M, 12k vocab
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=512,
            num_hidden_layers=16,
            num_attention_heads=8,
            num_key_value_heads=8,
            intermediate_size=2048,
            head_dim=64,
            max_window_layers=16,
            layer_types=['full_attention'] * 16,
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            hidden_act='silu',
            initializer_range=0.02,
            max_position_embeddings=2560,
            pad_token_id=self.tokenizer.pad_token_id,
            rms_norm_eps=1e-6,
            rope_parameters={
                'rope_theta': 1000000,
                'rope_type': 'default'
            },
            sliding_window=None,
            tie_word_embeddings=True,
            use_sliding_window=False,
            **model_config_kwargs,
        )

        self.model = Qwen3ForCausalLM(config=model_config)
        self.model.to(dtype=model_config.dtype)

        self.logger.append(f'>>> Done <<<', level='INFO', newline=True)
        # 统计模型参数量 （总参数量 & 可训练参数量，单位：Million）
        total_params = round(sum(p.numel() for p in self.model.parameters()) / 1_000_000, ndigits=2)
        trainable_params = round(sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1_000_000, ndigits=2)
        self.logger.info(f' - Total Parameters: {total_params} Million')
        self.logger.info(f' - Trainable Parameters: {trainable_params} Million')
        self.logger.info(f' - Model dtype: {self.model.dtype}')

        # 检查类型一致性
        if self.model.dtype != model_config.dtype:
            self.logger.critical(f'Model dtype ({self.model.dtype}) does not match config dtype ({model_config.dtype}).')
            raise ValueError(f'Model dtype ({self.model.dtype}) does not match config dtype ({model_config.dtype}).')

        if self.config.model.enable_grad_checkpointing:
            self.logger.info('Enable Gradient Checkpointing.')
            self.model.gradient_checkpointing_enable()

        if self.checkpoint and self.checkpoint.get('model', None) == self.model.__class__.__name__:
            self.logger.info('Loading model weights from checkpoint ... ', newline=False)
            self.model.load_state_dict(self.checkpoint['state']['weights'])
            self.logger.append(f'>>> Done <<<', level='INFO', newline=True)

        # 检查词嵌入层与分词器大小是否匹配
        if self.model.get_input_embeddings().weight.size(0) != len(self.tokenizer):
            self.logger.warning(f'Token embeddings size ({self.model.get_input_embeddings().weight.size(0)}) does not match tokenizer size ({len(self.tokenizer)}). Resizing token embeddings.')
            self.model.resize_token_embeddings(len(self.tokenizer))

        # 检查嵌入层是否绑定
        if model_config.tie_word_embeddings and self.model.get_input_embeddings().weight.data_ptr() != self.model.get_output_embeddings().weight.data_ptr():
            self.logger.critical('Model is configured to tie word embeddings, but input and output embeddings are not tied.')
            raise ValueError('Model is configured to tie word embeddings, but input and output embeddings are not tied.')

        if bool(self.config.training.freeze_embeddings):
            if self.TASK_TYPE == 'PreTraining':
                self.logger.warning('Freezing embedding layers during PreTraining may lead to suboptimal results.')
            self.logger.info('Freezing embedding layers ... ', newline=False)
            embedding = self.model.get_input_embeddings()
            for param in embedding.parameters():
                param.requires_grad = False
            self.logger.append(f'>>> Done <<<', level='INFO', newline=True)
            if self.model.get_input_embeddings().weight.data_ptr() == self.model.get_output_embeddings().weight.data_ptr():
                self.logger.info('Note: Input and output embeddings are tied. Output embeddings are also frozen.')

         # 冻结前 N 层
        freeze_layers = int(self.config.training.freeze_layers)
        if freeze_layers > 0:
            if self.TASK_TYPE == 'PreTraining':
                self.logger.warning('Freezing layers during PreTraining may lead to suboptimal results.')
            self.logger.info(f'Freezing the first {freeze_layers} layers ... ', newline=False)
            # 冻结前 N 层
            for layer in self.model.model.layers[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            self.logger.append(f'>>> Done <<<', level='INFO', newline=True)
        self.model.to(self.device)

        if self.is_distributed:
            self.model = DistributedDataParallel(
                module=self.model,
                device_ids=[self.device.index],
                output_device=self.device.index,
            )

    def init_model_probes(self):
        """Initialize model probes for monitoring internal states and gradients"""
        if self.is_main_process:
            self.model_probe = ModelProbe()
            num_hidden_layers = self.base_model.config.num_hidden_layers
            # 只监控第 1 层、中间层和最后一层
            self.model_probe.attach(model=self.base_model, module_names=[
                f'model.layers.0',  # 第 1 层
                f'model.layers.{num_hidden_layers // 2}',  # 中间层
                f'model.layers.{num_hidden_layers - 1}',  # 最后一层
            ])


if __name__ == '__main__':
    print('\033[0;32m ------------------------------\033[0m')
    print('\033[0;32m|          >>> Qi <<<          |\033[0m')
    print('\033[0;32m ------------------------------\033[0m')

    parser = argparse.ArgumentParser(description='预训练 Qwen3 模型')
    parser.add_argument('--nprocs', type=int, default=1, help='使用nproc个GPU进行训练')
    args = parser.parse_args()

    try:
        nprocs = args.nprocs
        if nprocs < 1: raise ValueError('nprocs must be at least 1.')
        elif nprocs == 1:
            trainer = Qwen3PreTrainer()
            trainer.start()
        elif nprocs > 1:
            import torch
            from torch import multiprocessing
            world_size = torch.cuda.device_count()
            if nprocs > world_size:
                raise ValueError(f'nprocs ({nprocs}) cannot be greater than available GPUs ({world_size}).')
            multiprocessing.spawn(
                fn=ddp_worker,
                args=(nprocs, Qwen3PreTrainer),
                nprocs=nprocs,
                join=True
            )
    except KeyboardInterrupt:
        print()
        print('\033[0;31m训练被打断。\033[0m')
