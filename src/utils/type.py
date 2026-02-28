import math
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Literal, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..logger import LogLevel
from ..module.model_presets import ModelSize


@dataclass(frozen=True, slots=True)
class ModelConfig:
    param_size: ModelSize = "0.3B"
    attn_implementation: Literal['eager', 'memory_efficient', 'sdpa', 'flash_attention_2', 'flash_attention_3'] = 'flash_attention_2'
    dtype: Literal['float32', 'bfloat16', 'float16'] = 'bfloat16'
    enable_grad_checkpointing: bool = False
    model_config_kwargs: Mapping[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AuxiliaryLossConfig:
    weight: float = 0.0
    recent_unlikelihood: dict = field(default_factory=lambda: {
        'window_size': 32,
        'max_neg_per_pos': 12,
    })


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int = 16  # 每个设备的批次大小
    learning_rate: float = 2.2e-4
    min_learning_rate: float = 2.2e-5
    optimizer_beta: tuple[float, float] = (0.9, 0.98)
    num_epochs: int = 1
    warmup_ratio: float = 0.018
    gradient_clipping_max_norm: float = 1.2
    quick_eval_per_epoch: int = 35  # 在每个 epoch 中进行多少次快速评估
    freeze_layers: int = 0  # 冻结前 n 层参数不进行训练
    freeze_embeddings: bool = False
    grad_accum_steps: int = 1  # 梯度累积步数
    auxiliary_loss: AuxiliaryLossConfig = field(default_factory=AuxiliaryLossConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        auxiliary_loss_config = config_dict.pop('auxiliary_loss')
        return cls(
            **config_dict,
            auxiliary_loss=AuxiliaryLossConfig(**auxiliary_loss_config)
        )


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    max_input_length: int = -1
    test_split: int | float = 0.05
    field_name: str = "tokenized_ids"


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    device: Literal['auto', 'cpu', 'cuda', 'mps'] = "auto"
    log_level: LogLevel = "INFO"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 36
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    notifications: dict[str, dict[str, str | int | float | bool]] = field(default_factory=dict)
    max_checkpoint_count: int = 18

    @staticmethod
    def check_config_integrity(config: dict, correct_config: dict | None = None) -> bool | list[str]:
        """ Check config integrity """
        if correct_config is None: correct_config = TrainerConfig().to_dict()
        missing = []
        for key, value in correct_config.items():
            if key not in config:
                missing.append(key)
                continue
            if isinstance(value, dict):
                if not isinstance(config[key], dict):
                    missing.append(key)  # Type mismatch
                else:
                    check_result = TrainerConfig.check_config_integrity(config[key], correct_config[key])
                    if check_result is not True:
                        for child_key in check_result:
                            missing.append(f'{key}.{child_key}')
        return True if not missing else missing

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainerConfig':
        """ Create TrainerConfig from a nested dictionary """
        missing_check = cls.check_config_integrity(config_dict)
        if missing_check is not True:
            print(f'Config file is missing the following fields: {[item for item in missing_check]}')
            exit(0)

        return cls(
            model=ModelConfig(**config_dict.get('model')),
            training=TrainingConfig.from_dict(config_dict.get('training')),
            dataset=DatasetConfig(**config_dict.get('dataset')),
            notifications=config_dict.get('notifications'),
            device=config_dict.get('device'),
            log_level=config_dict.get('log_level'),
            seed=config_dict.get('seed'),
            max_checkpoint_count=config_dict.get('max_checkpoint_count'),
        )

    def to_dict(self) -> dict:
        """ Convert TrainerConfig to a nested dictionary """
        return asdict(self)


@dataclass(slots=True)
class DatasetSize:
    train: int = 0
    valid: int = 0


@dataclass(slots=True)
class TrainerBuffer:
    dataset_size: DatasetSize = field(default_factory=DatasetSize)
    eval_steps: set[int] = field(default_factory=set)  # 每轮次第 eval_step 步时进行评估（在训练过程中进行评估的优化器步数集合）


@dataclass(frozen=True, slots=True)
class TrainEpochSize:
    micro_steps: int = 0  # 全局 micro-batch 数
    optimizer_steps: int = 0  # 大卡视角的 optimizer step 数


@dataclass(frozen=True, slots=True)
class ValidEpochSize:
    steps: int = 0


@dataclass(frozen=True, slots=True)
class EpochSize:
    train: TrainEpochSize = field(default_factory=TrainEpochSize)
    valid: ValidEpochSize = field(default_factory=ValidEpochSize)

    @classmethod
    def from_dict(cls, data_dict: dict[str, int | dict[str, int]]) -> 'EpochSize':
        return cls(
            train=TrainEpochSize(**data_dict.get('train')),
            valid=ValidEpochSize(**data_dict.get('valid'))
        )


@dataclass(slots=True)
class TrainerDataloader:
    train: DataLoader
    valid: DataLoader | None = None


@dataclass(slots=True)
class TrainMetricsAccumulator:
    loss: deque
    ppl: deque

    @property
    def avg_loss(self) -> float: return sum(self.loss) / len(self.loss) if self.loss else float('inf')
    @property
    def avg_ppl(self) -> float: return sum(self.ppl) / len(self.ppl) if self.ppl else float('inf')


@dataclass(frozen=True, slots=True)
class TrainStepOutput:
    logits: torch.Tensor
    loss: torch.Tensor
    auxiliary_loss: torch.Tensor | None = None

    labels: torch.LongTensor | None = None
    label_shift_mod: Literal['internal', 'external'] = 'internal'

    micro_step: int | None = None
    optimizer_step: int | None = None
    lr: float | None = None

    ignore_index: int = -100

    @property
    def ppl(self) -> float: return math.exp(self.loss.item())
    @property
    def auxiliary_ppl(self) -> float: return math.exp(self.auxiliary_loss.item()) if self.auxiliary_loss is not None else float('inf')
    @property
    def shift_labels(self) -> torch.LongTensor | None:
        if self.labels is None: return None
        if self.label_shift_mod == 'internal':
            labels = nn.functional.pad(self.labels, (0, 1), value=self.ignore_index)
            return labels[..., 1:].contiguous()
        elif self.label_shift_mod == 'external': return self.labels
        raise ValueError(f'Invalid label_shift_mod: {self.label_shift_mod}')


@dataclass
class VMemStatus:
    allocated: float
    reserved: float
    device_total_memory: float
    reserved_ratio: float
    usage_ratio: float
