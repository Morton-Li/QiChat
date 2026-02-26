import gc
import math
import os
import random
import time
from collections import deque
from contextlib import nullcontext
from typing import Literal

import numpy
import pandas
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim, autocast, distributed
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils import tensorboard
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer, get_scheduler, BatchEncoding, GenerationConfig

from .dataset import QiDataset
from .module import QiChatConfig, QiChatForCausalLM, MODEL_PRESETS
from .notifications import Notifier
from .probes import ModelProbe
from .tokenizer import QiTianTokenizerFast, get_tokenizer
from .tracker import TrainingTracker
from .utils.check import is_fork_supported, v_mem_status
from .utils.grad import collect_layer_grad_norms
from .utils.losses import build_recent_unlikelihood_negative_samples, compute_unlikelihood_loss_from_negatives
from .utils.masks import build_span_mask
from .utils.path import data_path, log_path
from .utils.type import TrainerConfig, TrainerBuffer, EpochSize, TrainerDataloader, TrainMetricsAccumulator
from .config import get_config
from .logger import Logger, get_logger


class TrainerBase:
    """ Base class for Trainer, providing common utilities and interfaces. """
    def __init__(self) -> None:
        self.config: TrainerConfig = get_config()
        self.logger: Logger = get_logger(log_filename=f'{self.__class__.__name__}.log', log_level=self.config.log_level)
        if not self.is_main_process: self.logger.disable()

        self.device: torch.device = self._get_device(device=self.config.device)
        if self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler()

        self.set_seed(seed=self.config.seed)

        self._buffer: TrainerBuffer = TrainerBuffer()

        if self.is_main_process:
            self._summary_writer = tensorboard.SummaryWriter(log_dir=str(log_path(self.__class__.__name__)))

    @property
    def is_distributed(self) -> bool:
        """Check if distributed training is initialized."""
        return distributed.is_available() and distributed.is_initialized()

    @property
    def is_main_process(self) -> bool:
        """Check if the current process is the main process in DDP."""
        if not self.is_distributed: return True
        return distributed.get_rank() == 0

    @property
    def micro_batch_size(self) -> int:
        """Calculate the micro-batch size per GPU."""
        return self.config.training.batch_size

    @property
    def step_batch_size(self) -> int:
        """Calculate the effective batch size for each training step, considering distributed training."""
        if self.is_distributed: return self.micro_batch_size * distributed.get_world_size()
        else: return self.micro_batch_size

    @property
    def optimizer_batch_size(self) -> int:
        """Calculate the effective batch size for each optimizer step."""
        return self.step_batch_size * self.config.training.grad_accum_steps

    def _get_device(self, device: Literal['auto', 'cuda', 'mps', 'cpu']) -> torch.device:
        """
        Get the device to run the model on.
        Args:
            device: Device to use, e.g., 'cpu', 'cuda', 'mps', or 'auto' to auto-detect.
        Returns:
            torch.device
        """
        if self.is_distributed:
            if device not in ['auto', 'cuda']:
                raise NotImplementedError('In DDP mode, only "auto" or "cuda" device selection is supported.')
            return torch.device(f'cuda:{distributed.get_rank()}')
        if device == 'auto':
            if torch.cuda.is_available():
                use_device = torch.device('cuda')
            elif torch.mps.is_available():
                use_device = torch.device("mps")
            else:
                use_device = torch.device("cpu")
        else:
            if device == "cuda" and torch.cuda.is_available():
                use_device = torch.device("cuda")
            elif device == "mps" and torch.mps.is_available():
                use_device = torch.device("mps")
            else:
                if device != "cpu":
                    self.logger.warning(f'Warning: Device {device} is not available. Falling back to CPU.')
                use_device = torch.device("cpu")
        return use_device

    def set_seed(self, seed: int = 36):
        """
        Set the random seed for reproducibility.
        :param seed: The seed value to set.
        """
        random.seed(seed)
        numpy.random.seed(seed=seed)
        torch.manual_seed(seed=seed)

    def set_dataset_size(self, train_samples: int, valid_samples: int):
        """ Set the dataset size """
        self._buffer.dataset_size.train = train_samples
        self._buffer.dataset_size.valid = valid_samples

    @property
    def epoch_size(self) -> EpochSize:
        """ Get the epoch size information """
        return EpochSize.from_dict({
            'train': {
                'micro_steps': math.ceil(self._buffer.dataset_size.train / self.micro_batch_size),  # 每个 epoch 的 micro-batch 数
                'optimizer_steps': math.ceil(self._buffer.dataset_size.train / self.optimizer_batch_size)
            },
            'valid': {
                'steps': math.ceil(self._buffer.dataset_size.valid / self.micro_batch_size)
            },
        })

    @property
    def summary_writer(self) -> tensorboard.SummaryWriter:
        if self.is_main_process: return self._summary_writer
        # 返回一个假函数防止在非主进程中调用 summary_writer 导致错误
        else: return lambda *args, **kwargs: None


class Trainer(TrainerBase):
    """ QiChat Trainer """
    VERSION = '1.6.0'
    TASK_TYPE: Literal['PreTraining', 'FineTuning']

    def __init__(self, *args, **kwargs):
        super().__init__()
        if self.is_distributed and self.is_main_process:
            self.logger.info(f'Distributed training is enabled. world_size: {distributed.get_world_size()}')

        self.logger.info(f'{"Auto-detected" if self.config.device == "auto" else "Using"} device: {self.device.type}')

        self.model: nn.Module | DistributedDataParallel
        self.checkpoint: dict = {}
        self.tokenizer: PreTrainedTokenizer
        self.optimizer: optim.Optimizer
        self.criterion: nn.Module
        # self.scheduler: LayerWiseDummyScheduler | optim.lr_scheduler.LambdaLR | optim.lr_scheduler.ReduceLROnPlateau
        self.data_loader: TrainerDataloader
        self.training_tracker: TrainingTracker
        self.model_probe: ModelProbe
        self.notifier: Notifier

        self.load_checkpoint()
        self.init_tokenizer()
        self.init_model()
        self.init_optimizer()
        self.init_loss_fn()
        self.init_auxiliary_loss()
        self.init_training_tracker()
        self.init_model_probes()
        self.init_notifier()

        self.load_dataset()
        self.init_scheduler()

    def load_checkpoint(self):
        """Load the model checkpoint."""
        checkpoint_path = data_path('checkpoint')
        # CheckPoint_{model_name}-{param_size}_{epoch}.{batch}_{total_batch}_{yyyy-mm-dd}.pth
        checkpoint_list = [checkpoint for checkpoint in os.listdir(checkpoint_path) if checkpoint.startswith('CheckPoint_') and checkpoint.endswith('.pth')]

        if not checkpoint_list:
            if self.TASK_TYPE == 'FineTuning':
                self.logger.error(f'No checkpoint files found. {self.TASK_TYPE} requires a pre-trained model checkpoint.')
                raise FileNotFoundError(f'No checkpoint files found. {self.TASK_TYPE} requires a pre-trained model checkpoint.')
            setattr(self, 'checkpoint', {})
            return

        # 自定义排序函数
        def extract_sort_key(filename):
            import re
            match = re.search(r'_(\d+\.\d+)_(\d+)_', filename)
            if match:
                epoch_batch_num = float(match.group(1))  # 7.800 -> 7.800
                total_batch = int(match.group(2))  # 30260 -> 30260
                return total_batch
            return 0

        checkpoint_list.sort(key=extract_sort_key)

        self.logger.info('Loading the checkpoint ... \t', newline=False)
        load_checkpoint_file = checkpoint_list[-1]
        checkpoint = torch.load(os.path.join(checkpoint_path, load_checkpoint_file), map_location=self.device)
        self.checkpoint = checkpoint
        self.logger.append(f'>>> Done <<<', level='INFO', newline=True)

        trainer_version = checkpoint['trainer']['version'] if 'trainer' in checkpoint and 'version' in checkpoint['trainer'] else 'unknown'
        if self.TASK_TYPE == checkpoint['type'] and trainer_version != self.VERSION:
            self.logger.warning(f'Checkpoint trainer version ({trainer_version}) does not match current trainer version ({self.VERSION}). This may lead to compatibility issues.')
        self.logger.info(f'Checkpoint loaded from {load_checkpoint_file}')

        if not self.is_distributed:
            # 避免冲突将所有checkpoint_list文件增加.bak后缀
            for checkpoint in checkpoint_list:
                os.rename(
                    os.path.join(checkpoint_path, checkpoint),
                    os.path.join(
                        checkpoint_path,
                        f'{checkpoint}.load' if checkpoint == load_checkpoint_file else f'{checkpoint}.bak'
                    )
                )

    def init_tokenizer(self):
        """Initialize the QiTokenizer"""
        self.tokenizer: QiTianTokenizerFast = get_tokenizer()

    def init_model(self):
        """Initialize the QiChat Model"""
        self.logger.info('Initializing QiChat Model ... ', newline=False)
        if self.config.model.param_size not in MODEL_PRESETS.keys():
            self.logger.critical(f'Unsupported model param_size: {self.config.model.param_size}')
            raise ValueError(f'Unsupported model param_size: {self.config.model.param_size}')
        # 根据 param_size 设置模型配置参数
        model_config_kwargs = MODEL_PRESETS[self.config.model.param_size].copy()
        model_config_kwargs['attn_implementation'] = self.config.model.attn_implementation
        model_config_kwargs['dtype'] = self.config.model.dtype
        model_config_kwargs.update(self.config.model.model_config_kwargs)

        model_config = QiChatConfig(**model_config_kwargs)
        self.model = QiChatForCausalLM(config=model_config)
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
            for block in self.model.model.blocks[:freeze_layers]:
                for param in block.parameters():
                    param.requires_grad = False
            self.logger.append(f'>>> Done <<<', level='INFO', newline=True)
        self.model.to(self.device)

        if self.is_distributed:
            self.model = DistributedDataParallel(
                module=self.model,
                device_ids=[self.device.index],
                output_device=self.device.index,
            )

    @property
    def base_model(self) -> nn.Module:
        """Get the base model, handling DDP if necessary."""
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

    def init_optimizer(self):
        """Initialize the optimizer"""
        decay_params = []
        no_decay_params = []

        # 收集所有 LayerNorm 模块下的参数名
        layernorm_names = set()
        embedding_names = set()
        for module_name, module in self.base_model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                for param_name, _ in module.named_parameters():
                    full_name = f"{module_name}.{param_name}" if module_name else param_name
                    layernorm_names.add(full_name)
            if isinstance(module, nn.Embedding):
                for param_name, _ in module.named_parameters():
                    full_name = f"{module_name}.{param_name}" if module_name else param_name
                    embedding_names.add(full_name)

        # 精准分组参数
        seen_params = set()
        for name, param in self.base_model.named_parameters():
            if not param.requires_grad or param in seen_params:
                continue
            if (name in layernorm_names) or (name in embedding_names) or name.endswith("bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
            seen_params.add(param)

        self.optimizer = optim.AdamW(params=[
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=self.config.training.learning_rate, betas=self.config.training.optimizer_beta)

        # 确认参数是否被遗漏
        all_param_set = set(p for g in self.optimizer.param_groups for p in g["params"])
        for name, param in list(self.base_model.named_parameters()):
            if not param.requires_grad:
                continue  # 忽略不参与训练的参数
            if param not in all_param_set:
                self.logger.error(f"Parameter not in optimizer: {name}")
                raise RuntimeError(f"Parameter not in optimizer: {name}")

        if self.checkpoint and self.checkpoint['type'] == self.TASK_TYPE and self.checkpoint.get('model', None) == self.model.__class__.__name__:
            self.logger.info('Loading optimizer state from checkpoint ... ', newline=False)
            self.optimizer.load_state_dict(self.checkpoint['state']['optimizer'])
            self.logger.append(f'>>> Done <<<', level='INFO', newline=True)
        else:
            self.logger.info(f'Optimizer: {self.optimizer.__class__.__name__}, Learning rate: {self.config.training.learning_rate} (min_lr: {self.config.training.min_learning_rate})')

    def init_loss_fn(self):
        """Initialize the loss function"""
        pass

    def init_auxiliary_loss(self):
        """Initialize auxiliary loss functions if needed"""
        # TODO: 需要添加辅助损失函数
        pass

    def init_scheduler(self):
        """
        Initialize the learning rate scheduler
        :return:
        """
        num_warmup_steps = math.ceil(self.epoch_size.train.optimizer_steps * self.config.training.warmup_ratio)

        self.scheduler = get_scheduler(
            name='linear',
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=int(self.epoch_size.train.optimizer_steps * self.config.training.num_epochs),
        )
        self.logger.info(f'Warmup steps: {num_warmup_steps} ({self.config.training.warmup_ratio:.2%} of one epoch)')

        if self.checkpoint and self.checkpoint['type'] == self.TASK_TYPE and self.checkpoint.get('model', None) == self.model.__class__.__name__:
            self.logger.info('Loading scheduler state from checkpoint ... ', newline=False)
            self.scheduler.load_state_dict(self.checkpoint['state']['scheduler'])
            self.logger.append(f'>>> Done <<<', level='INFO', newline=True)

    @property
    def is_warmup(self) -> bool:
        """Check if currently in the warmup phase."""
        current_step = self.training_tracker.batch.total_batches
        num_warmup_steps = math.ceil(self.epoch_size.train.optimizer_steps * self.config.training.warmup_ratio)
        return current_step < num_warmup_steps

    def unpack_dataset(self, dataset: pandas.DataFrame) -> tuple[list[numpy.ndarray], list[numpy.ndarray]]:
        """Unpack dataset into inputs and labels"""
        raise NotImplementedError('unpack_dataset method must be implemented in subclasses.')

    def load_dataset(self):
        """Load the training and validation datasets"""
        self.logger.info('Loading dataset ... ', newline=False)

        dataset_file_list = list(data_path('dataset').glob('**/*.parquet'))
        if not dataset_file_list:
            self.logger.append(message='>>> Dataset not found <<<', level='CRITICAL', newline=True)
            raise FileNotFoundError('Dataset not found')

        # 拼接dataset_file数据集
        dataset_df = pandas.concat([pandas.read_parquet(file) for file in dataset_file_list], ignore_index=True)
        self.logger.append(message=f'>>> Loaded {len(dataset_df)} records on {len(dataset_file_list)} files <<<', level='INFO', newline=True)

        # 删除过长的样本
        if self.config.dataset.max_input_length > 0:
            original_length = len(dataset_df)
            dataset_df = dataset_df[(dataset_df[self.config.dataset.field_name].apply(len) <= self.config.dataset.max_input_length)].reset_index(drop=True)
            if len(dataset_df) != original_length:
                self.logger.warning(f'Deleted {original_length - len(dataset_df)} samples that are too long')

        # 根据目标字段构建input和label
        inputs, labels = self.unpack_dataset(dataset=dataset_df)
        del dataset_df

        # 划分训练集和验证集
        if self.config.dataset.test_split <= 0:
            self.logger.warning('No validation dataset will be used.')
            x_train, y_train = inputs, labels
            x_test, y_test = pandas.Series([], dtype=object), pandas.Series([], dtype=object)
        else:
            self.logger.info('Splitting training and validation datasets ...')
            x_train, x_test, y_train, y_test = train_test_split(
                inputs, labels,
                test_size=self.config.dataset.test_split,
                shuffle=True,
                random_state=self.config.seed,
            )
        gc.collect()

        self.set_dataset_size(train_samples=len(x_train), valid_samples=len(x_test))

        if self.config.training.quick_eval_per_epoch > 0:
            eval_steps: set[int] = {
                round(self.epoch_size.train.optimizer_steps * i / (self.config.training.quick_eval_per_epoch + 1))
                for i in range(1, self.config.training.quick_eval_per_epoch + 1)  # 1 ... quick_eval_per_epoch
            }
            eval_steps.discard(0)
            eval_steps.discard(self.epoch_size.train.optimizer_steps)
            self._buffer.eval_steps = eval_steps

        self.logger.info(f'Loaded dataset: {self._buffer.dataset_size.train} training samples, {self._buffer.dataset_size.valid} validation samples')

        if not isinstance(self.config.training.grad_accum_steps, int):
            self.logger.critical('grad_accum_steps must be an integer.')
            raise ValueError('grad_accum_steps must be an integer.')
        if self.config.training.grad_accum_steps < 1:
            self.logger.critical('grad_accum_steps must be at least 1.')
            raise ValueError('grad_accum_steps must be at least 1.')

        # 检查批次大小
        if self.config.training.batch_size < 1:
            self.logger.critical('Batch size must be greater than 0.')
            raise ValueError('Batch size must be greater than 0.')
        if not isinstance(self.config.training.batch_size, int):
            self.logger.critical('Batch size must be an integer.')
            raise ValueError('Batch size must be an integer.')

        self.logger.info(f'Epoch size: {self.epoch_size.train.optimizer_steps}.')
        # 计算等效批次大小
        if self.is_distributed:
            if self.config.training.grad_accum_steps > 1:
                self.logger.info(f' - Global step batch size: {self.step_batch_size}')
                self.logger.info(f' - Gradient accumulation steps: {self.config.training.grad_accum_steps}')
                self.logger.info(f' - Optimizer step batch size: {self.optimizer_batch_size}')
            else:
                self.logger.info(f' - Batch size: {self.step_batch_size}')
            self.logger.info(f' - Per-GPU step size: {self.micro_batch_size}')
        else:
            if self.config.training.grad_accum_steps > 1:
                self.logger.info(f' - Step batch size: {self.step_batch_size}')
                self.logger.info(f' - Gradient accumulation steps: {self.config.training.grad_accum_steps}')
                self.logger.info(f' - Optimizer step batch size: {self.optimizer_batch_size}')
            else:
                self.logger.info(f' - Batch size: {self.step_batch_size}')

        self.logger.info('Loading DataLoader ... ', newline=False)
        # 记录开始时间
        load_start_time = time.time()

        train_dataset = QiDataset(input_ids=list(x_train), label_ids=list(y_train))
        valid_dataset = QiDataset(input_ids=list(x_test), label_ids=list(y_test))
        train_sampler = DistributedSampler(
            dataset=train_dataset,
            shuffle=True,
        ) if self.is_distributed else None

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=train_sampler,
            collate_fn=self.batch_preprocess,
            pin_memory=True if self.device.type == 'cuda' else False,
            num_workers=min(2, os.cpu_count() // 2) if is_fork_supported() else 0,  # 取 CPU 核心数的一半，但不超过 2
            persistent_workers=is_fork_supported(),
            shuffle=not self.is_distributed,
        )
        valid_dataloader = None
        if self.is_main_process:
            valid_dataloader = DataLoader(
                dataset=valid_dataset,
                batch_size=self.config.training.batch_size,
                collate_fn=self.batch_preprocess,
                pin_memory=True if self.device.type == 'cuda' else False,
                num_workers=min(2, os.cpu_count() // 2) if is_fork_supported() else 0,  # 取 CPU 核心数的一半，但不超过 2
                persistent_workers=is_fork_supported(),
                shuffle=False
            )
        self.data_loader = TrainerDataloader(train=train_dataloader, valid=valid_dataloader,)
        self.logger.append(f'>>> Done <<< (in {time.time() - load_start_time:.2f} seconds)', level='INFO', newline=True)

    def batch_preprocess(self, batch: list[tuple[torch.LongTensor, torch.LongTensor]]) -> tuple[BatchEncoding, torch.Tensor]:
        """
        Preprocess the batch
        :param batch:
        :return:
        """
        input_ids, label_ids = zip(*batch)

        input_ids = pad_sequence(sequences=input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        label_ids = pad_sequence(sequences=label_ids, batch_first=True, padding_value=-100)

        return BatchEncoding({
            'input_ids': input_ids,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id)  # 生成注意力掩码
        }), label_ids

    def init_training_tracker(self):
        """Initialize the training tracker"""
        self.training_tracker = TrainingTracker()

        # 从检查点加载数据
        if self.checkpoint and self.checkpoint['type'] == self.TASK_TYPE:
            self.training_tracker.epoch.set(value=self.checkpoint['progress']['epoch'])
            # 训练不会从断点开始所以不恢复批次数据
            # self.training_tracker.batch.set(value=self.checkpoint['progress']['batch'])
            self.training_tracker.batch.set_total_batches(value=self.checkpoint['progress']['total_batches'])
            self.training_tracker.loss.valid._min = self.checkpoint['progress']['loss']

    def init_model_probes(self):
        """Initialize model probes for monitoring internal states and gradients"""
        if self.is_main_process:
            self.model_probe = ModelProbe()
            num_hidden_layers = self.base_model.config.num_hidden_layers
            # 只监控第 1 层、中间层和最后一层
            self.model_probe.attach(model=self.base_model, module_names=[
                f'model.blocks.0',  # 第 1 层
                f'model.blocks.{num_hidden_layers // 2}',  # 中间层
                f'model.blocks.{num_hidden_layers - 1}',  # 最后一层
            ])

    def _log_model_probe_to_tb(self, prefix: str | None = None) -> None:
        """Log model probe data to TensorBoard"""
        if not self.is_main_process or not self.model_probe.active: return
        model_probe_result: dict[str, dict[str, int | float | dict[str, int | float]]] = self.model_probe.flattened_results
        for metric_name, dict_data in model_probe_result.items():
            # metric_name in ['representation_diversity_metrics', 'output_distribution_metrics']
            # dict_data 全为 dict
            for key, value in dict_data.items():
                if isinstance(value, dict):
                    self.summary_writer.add_scalars(
                        main_tag=f'{prefix}/{key}',
                        tag_scalar_dict=value,
                        global_step=self.training_tracker.batch.total_batches
                    )
                elif isinstance(value, (int, float)):
                    self.summary_writer.add_scalar(
                        tag=f'{prefix}/{key}',
                        scalar_value=value,
                        global_step=self.training_tracker.batch.total_batches
                    )

    def init_notifier(self):
        """ Initialize the notifier """
        if self.is_main_process and self.config.notifications:
            self.logger.info('Notifications enabled.')
            self.notifier = Notifier.from_config(config=self.config.notifications)
        else: self.notifier = None

    def save_model(self):
        """
        Save the model
        :return:
        """
        if not self.is_main_process: return
        self.model.eval()

        pth_dict = {
            'model': self.base_model.__class__.__name__,
            'description': '',
            'type': self.TASK_TYPE,
            'state': {
                'weights': self.base_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            },
            'progress': {
                'epoch': self.training_tracker.epoch(),
                'batch': self.training_tracker.batch(),
                'total_batches': self.training_tracker.batch.total_batches,
                'loss': self.training_tracker.loss.valid.avg,
            },
            'dataset': {
                'total_samples': self._buffer.dataset_size.train + self._buffer.dataset_size.valid,
                'train_samples': self._buffer.dataset_size.train,
                'valid_samples': self._buffer.dataset_size.valid,
            },
            'trainer': {
                'version': self.VERSION,
            },
            'author': 'Morton Li',
            'timestamp': int(time.time()),
        }

        file_prefix = f'CheckPoint_{self.base_model.__class__.__name__}-{self.config.model.param_size}_{self.TASK_TYPE}_'
        checkpoint_list = [
            checkpoint
            for checkpoint in os.listdir(data_path('checkpoint'))
            if checkpoint.startswith(file_prefix) and checkpoint.endswith('.pth')
        ]
        # 按照文件修改时间排序
        checkpoint_list.sort(key=lambda x: os.path.getmtime(data_path('checkpoint', x)))

        file_name = f'{file_prefix}{self.training_tracker.epoch()}.{self.training_tracker.batch()}_{self.training_tracker.batch.total_batches}_{time.strftime("%Y-%m-%d", time.localtime())}.pth'
        try:
            torch.save(pth_dict, data_path('checkpoint', file_name))
        except (OSError, RuntimeError) as e:
            # 删除 checkpoint_list 最早的一个文件后重试
            if checkpoint_list:
                os.remove(data_path('checkpoint', checkpoint_list[0]))
                checkpoint_list.pop(0)
                torch.save(pth_dict, data_path('checkpoint', file_name))
            else:
                self.logger.critical(f'Failed to save checkpoint: {e}')
                raise e

        num_checkpoints = len(checkpoint_list) + 1
        if num_checkpoints > self.config.max_checkpoint_count:
            # 计算差值
            diff = num_checkpoints - self.config.max_checkpoint_count
            # 删除最早的文件
            for i in range(diff):
                os.remove(data_path('checkpoint', checkpoint_list[i]))

    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute the loss
        Returns:
            torch.Tensor: Loss tensor
        """
        raise NotImplementedError('compute_loss method must be implemented in subclasses.')

    def compute_auxiliary_loss(
        self,
        logits: torch.Tensor,
        labels: torch.LongTensor,
        shift_mod: Literal['internal', 'external'] = 'internal',
    ) -> torch.Tensor:
        """
        Compute auxiliary loss if needed
        Returns:
            torch.Tensor: Auxiliary loss tensor
        """
        loss = torch.tensor(0.0, device=self.device)
        if self.config.training.auxiliary_loss.weight <= 0: return loss

        if shift_mod == 'internal':
            # 内部 shift，需要同步调整 labels 以对齐模型内部的 shift 方式
            labels = nn.functional.pad(labels, (0, 1), value=-100)
            labels = labels[..., 1:].contiguous()

        recent_ul_window_size = self.config.training.auxiliary_loss.recent_unlikelihood['window_size']
        recent_ul_max_neg_per_pos = self.config.training.auxiliary_loss.recent_unlikelihood['max_neg_per_pos']
        if recent_ul_window_size > 0 and recent_ul_max_neg_per_pos > 0:
            pos_flat, neg_ids = build_recent_unlikelihood_negative_samples(
                input_ids=labels,
                window_size=recent_ul_window_size,
                max_neg_per_pos=recent_ul_max_neg_per_pos,
                ignore_token_ids={
                    self.tokenizer.pad_token_id, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, -100
                },
                chunk=512,
            )

            loss += compute_unlikelihood_loss_from_negatives(
                predictions=logits,
                neg_pos=pos_flat, neg_ids=neg_ids,
            )

        return self.config.training.auxiliary_loss.weight * loss

    def clip_grads(self):
        """
        Clip gradients if necessary and log clipping stats
        """
        if self.config.training.gradient_clipping_max_norm <= 0: return
        raw_total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.training.gradient_clipping_max_norm)

        if self.is_main_process:
            raw_total_norm = float(raw_total_norm)
            if not math.isfinite(raw_total_norm):  # 非有限值：强制标记为已裁剪，scale 记 0.0（力度极强/异常）
                is_clipped, clip_scale = True, 0.0
            else:
                is_clipped = raw_total_norm > self.config.training.gradient_clipping_max_norm
                clip_scale = min(1.0, self.config.training.gradient_clipping_max_norm / (raw_total_norm + 1e-6))
            self.training_tracker.grad_clip_meter.step(is_clipped=is_clipped, clip_scale=clip_scale)

    def step(self, batch_idx: int, inputs: BatchEncoding, labels: torch.LongTensor | None = None) -> tuple[torch.Tensor, float, float, float]:
        """
        One training/evaluation step
        Args:
            batch_idx (int): Batch index
            inputs (BatchEncoding): Input batch
            labels (Optional[torch.LongTensor]): Label batch
        Returns:
            tuple[torch.Tensor, float, float, float]: Logits, loss, perplexity, auxiliary loss
        """
        if self.model.training:
            # 每个 rank 实际能看到的 micro-batch 数
            local_epoch_micro = len(self.data_loader.train)
            current_micro_step = batch_idx + 1
            is_last_batch = (current_micro_step >= local_epoch_micro)
            accum_steps = self.config.training.grad_accum_steps if not is_last_batch else (current_micro_step % self.config.training.grad_accum_steps or self.config.training.grad_accum_steps)

            if batch_idx % self.config.training.grad_accum_steps == 0:  # 每 grad_accum_steps 的第一步清除梯度
                self.optimizer.zero_grad()  # 清除梯度

            is_opt_step = (current_micro_step % self.config.training.grad_accum_steps == 0 or is_last_batch)
            if is_opt_step:
                # 这里的 total_batches 就是“优化器步”的计数
                self.training_tracker.batch.next()
            current_step_lr = self.optimizer.param_groups[0]['lr']

            ddp_context = (self.model.no_sync() if self.is_distributed and not is_opt_step else nullcontext())

        # 混合精度训练
        if self.device.type == 'cuda' and getattr(self, 'scaler', None) is not None:
            with autocast(device_type=self.device.type, dtype=self.base_model.config.dtype):
                outputs = self.model(**inputs)

                if labels is not None:
                    loss = self.compute_loss(predictions=outputs.logits, labels=labels)
                elif outputs.loss is None:
                    self.logger.critical('Labels are None but model output does not contain loss.')
                    raise ValueError('Labels are None but model output does not contain loss.')
                else:
                    loss = outputs.loss

                # 记录原始损失
                original_loss = loss
                if self.config.training.auxiliary_loss.weight > 0:
                    loss = loss + self.compute_auxiliary_loss(
                        logits=outputs.logits,
                        labels=labels if labels is not None else inputs.labels,
                        shift_mod='internal' if labels is None else 'external',
                    )  # 如果有辅助损失则加上

                # 反向传播和权重更新
                if self.model.training:
                    scaled_loss = loss / accum_steps  # 梯度累积时需要除以累积步数
                    with ddp_context:
                        if self.base_model.config.dtype in [torch.bfloat16]: scaled_loss.backward()
                        else: self.scaler.scale(scaled_loss).backward()

                    if is_opt_step:
                        # 梯度裁剪
                        if self.config.training.gradient_clipping_max_norm > 0:
                            # 不走 scaler.scale 的路径需要手动 unscale 才能裁剪
                            if self.base_model.config.dtype not in [torch.bfloat16]: self.scaler.unscale_(self.optimizer)
                            self.clip_grads()

                        if self.base_model.config.dtype in [torch.bfloat16]: self.optimizer.step()
                        else:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        # 如果当前处于 warmup 阶段或者学习率没有降到最低值以下，才更新学习率
                        if self.is_warmup or current_step_lr > self.config.training.min_learning_rate:
                            self.scheduler.step()
        else:
            outputs = self.model(**inputs)

            if labels is not None:
                loss = self.compute_loss(predictions=outputs.logits, labels=labels)
            elif outputs.loss is None:
                self.logger.critical('Labels are None but model output does not contain loss.')
                raise ValueError('Labels are None but model output does not contain loss.')
            else:
                loss = outputs.loss

            # 记录原始损失
            original_loss = loss
            if self.config.training.auxiliary_loss.weight > 0:
                loss = loss + self.compute_auxiliary_loss(
                    logits=outputs.logits,
                    labels=labels if labels is not None else inputs.labels,
                    shift_mod='internal' if labels is None else 'external',
                )  # 如果有辅助损失则加上

            # 反向传播和权重更新
            if self.model.training:
                scaled_loss = loss / accum_steps  # 梯度累积时需要除以累积步数
                with ddp_context:
                    scaled_loss.backward()

                if is_opt_step:
                    # 梯度裁剪
                    if self.config.training.gradient_clipping_max_norm > 0:
                        self.clip_grads()
                    self.optimizer.step()
                    if self.is_warmup or current_step_lr > self.config.training.min_learning_rate:
                        self.scheduler.step()

        ppl = math.exp(original_loss.item())

        if self.is_main_process and self.model_probe.active:
            self.model_probe.compute_output_distribution_metrics(
                logits=outputs.logits,
                labels=labels if labels is not None else inputs.labels,
                topk=5,
                ignore_index=-100,
                shift_mod='external' if labels is not None else 'internal',
                max_tokens=128,
            )
            self.model_probe.compute_representation_diversity_metrics(
                attention_mask=inputs.attention_mask,
                max_tokens=128,
            )

        return outputs.logits.detach(), original_loss.detach().item(), ppl, loss.detach().item()

    @torch.no_grad()
    def inference_samples(self):
        """
        Inference some samples
        """
        pass

    def evaluate_epoch(self, quick_test: bool = True):
        """
        Evaluate for one epoch
        Args:
            quick_test: Whether to perform a quick test (only config.batch_size batches)
        """
        if not self.is_main_process: return
        if self._buffer.dataset_size.valid <= 0 or not hasattr(self.data_loader, 'valid'):
            # self.logger.warning('No validation dataset found, skipping evaluation.')
            self.save_model()
            self.inference_samples()
            return

        sample_total = len(self.data_loader.valid)
        if quick_test: sample_total = min(max(self.config.training.batch_size * 10, 64), sample_total)

        self.model.eval()

        self.model_probe.set_active(active=True)  # 评估时开启模型探针监控内部状态和梯度

        # 等待1秒确保所有的输出缓冲区均已刷新
        time.sleep(1)

        total_ppl = 0

        with torch.no_grad(), tqdm(
            total=sample_total,
            unit='batch',
            desc=f'Evaluating',
            dynamic_ncols=True,
            leave=False
        ) as pbar:
            for batch_idx, (input_batch, label_batch) in enumerate(self.data_loader.valid):
                if label_batch is not None:
                    label_batch = label_batch.to(self.device)
                try:
                    logits, batch_loss, ppl, aux_loss = self.step(
                        batch_idx=batch_idx,
                        inputs=input_batch.to(self.device),
                        labels=label_batch,
                    )

                    # 更新验证损失
                    self.training_tracker.loss.valid.step(loss=batch_loss)
                    total_ppl += ppl

                except torch.OutOfMemoryError as oom_error:
                    self.logger.warning('Out of memory error occurred.')
                    if self.is_distributed:
                        # OOM recovery is not supported in DDP mode.
                        print(f'[rank {distributed.get_rank()}] Raising OOM error in DDP mode during evaluation.')
                        raise oom_error
                    if not self.model.is_gradient_checkpointing:
                        self.logger.info('Enabling gradient checkpointing to reduce memory usage.')
                        self.model.gradient_checkpointing_enable()
                    else:
                        self.logger.error('Model is already using gradient checkpointing. Cannot recover from OOM.')
                        raise oom_error
                    continue
                finally:
                    # 即使continue我们也需要更新进度条
                    pbar.update(1)

                if self.model_probe.active:
                    # 及时处理模型探针
                    self._log_model_probe_to_tb(prefix='Valid/batch')
                    self.model_probe.set_active(active=False)

                if quick_test and (batch_idx + 1) >= sample_total: break

        # 记录到TensorBoard
        self.summary_writer.add_scalars(
            main_tag='Valid/batch/Loss',
            tag_scalar_dict={
                'Avg': self.training_tracker.loss.valid.avg,
                'Best': self.training_tracker.loss.valid.min,
                'Worst': self.training_tracker.loss.valid.max,
            },
            global_step=self.training_tracker.batch.total_batches
        )
        self.summary_writer.add_scalar(
            tag='Valid/batch/AvgPPL',
            scalar_value=round(total_ppl / sample_total, 4),
            global_step=self.training_tracker.batch.total_batches
        )

        if self.training_tracker.loss.valid.is_best or quick_test is False:
            self.save_model()
            self.training_tracker.loss.valid.reset_best()

        self.inference_samples()

        self.training_tracker.loss.valid.reset()

    def train_epoch(self):
        """ Train for one epoch """
        if self.training_tracker.epoch() == 0 and self.training_tracker.batch.total_batches == 0:
            # 训练前先进行一次完整评估
            self.evaluate_epoch(quick_test=False)
            # 整个训练的第一步开启模型探针看模型最初状态
            if self.is_main_process: self.model_probe.set_active(active=True)

        if self.is_distributed:  # DDP 需要在每个 epoch 开始时调用 set_epoch 来确保数据被正确分配
            self.data_loader.train.sampler.set_epoch(self.training_tracker.epoch())
        self.training_tracker.epoch.next()

        self.model.train()

        # 优化器步训练指标累加器
        train_metrics_accumulator: TrainMetricsAccumulator = TrainMetricsAccumulator(
            loss=deque(maxlen=max(1, self.config.training.grad_accum_steps)),
            ppl=deque(maxlen=max(1, self.config.training.grad_accum_steps)),
        )

        # 等待1秒确保所有的输出缓冲区均已刷新
        time.sleep(1)

        # 记录用于计算每秒训练步数的时间
        speed_time = time.time()

        with tqdm(
            total=self.epoch_size.train.optimizer_steps,
            unit='step',
            desc=f'Training - Epoch {self.training_tracker.epoch()}',
            postfix={'Loss': 0, 'LR': f'{self.optimizer.param_groups[0]['lr']:.6f}'},
            dynamic_ncols=True,
            leave=False,
            disable=not self.is_main_process,
        ) as pbar:
            for batch_idx, (input_batch, label_batch) in enumerate(self.data_loader.train):
                before_step_lr = self.optimizer.param_groups[0]['lr']
                micro_step = batch_idx + 1
                is_opt_step = (self.config.training.grad_accum_steps == 1) or (micro_step % self.config.training.grad_accum_steps == 0) or (micro_step == len(self.data_loader.train))
                if is_opt_step:
                    opt_step = micro_step // self.config.training.grad_accum_steps  # 计算优化器步

                    # 每 500 优化器步
                    if self.is_main_process and opt_step % 500 == 0:
                        self.model_probe.set_active(active=True)

                if label_batch is not None:
                    label_batch = label_batch.to(self.device, non_blocking=True if self.data_loader.train.pin_memory else False)
                try:
                    logits, batch_loss, ppl, aux_loss = self.step(
                        batch_idx=batch_idx,
                        inputs=input_batch.to(self.device, non_blocking=True if self.data_loader.train.pin_memory else False),
                        labels=label_batch,
                    )

                    # 更新累积步映射
                    train_metrics_accumulator.loss.append(batch_loss)
                    train_metrics_accumulator.ppl.append(ppl)
                except torch.OutOfMemoryError as oom_error:
                    self.logger.warning('Out of memory error occurred.')
                    if self.is_distributed:
                        # OOM recovery is not supported in DDP mode.
                        print(f'[rank {distributed.get_rank()}] Raising OOM error in DDP mode.')
                        if self.is_main_process and getattr(self, 'notifier', None) is not None:
                            self.notifier.notify(
                                subject='OOM Error in DDP Mode',
                                content=f'An out of memory error occurred on rank {distributed.get_rank()} during training. OOM recovery is not supported in DDP mode.'
                            )
                        raise oom_error
                    else:
                        if self.config.training.grad_accum_steps > 1:
                            # 在启用梯度累积的情况下无法确保 step 函数内执行 optimizer.zero_grad()
                            self.logger.critical(f'Cannot recover from OOM when gradient accumulation is enabled (grad_accum_steps={self.config.training.grad_accum_steps}).')
                            if self.is_main_process and getattr(self, 'notifier', None) is not None:
                                self.notifier.notify(
                                    subject='OOM Error during Training',
                                    content=f'An out of memory error occurred during training. Cannot recover when gradient accumulation is enabled (grad_accum_steps={self.config.training.grad_accum_steps}).'
                                )
                            raise oom_error
                        if not self.model.is_gradient_checkpointing:
                            self.logger.info('Enabling gradient checkpointing to reduce memory usage.')
                            self.model.gradient_checkpointing_enable()
                            if self.is_main_process and getattr(self, 'notifier', None) is not None:
                                self.notifier.notify(
                                    subject='Gradient Checkpointing Enabled',
                                    content='An out of memory error occurred during training. Gradient checkpointing has been enabled to reduce memory usage.'
                                )
                            continue
                        else:
                            self.logger.error('Model is already using gradient checkpointing. Cannot recover from OOM.')
                            if self.is_main_process and getattr(self, 'notifier', None) is not None:
                                self.notifier.notify(
                                    subject='OOM Error during Training',
                                    content='An out of memory error occurred during training. Model is already using gradient checkpointing and cannot recover from OOM.'
                                )
                            raise oom_error
                finally:
                    if is_opt_step: pbar.update(n=1)

                if self.optimizer.param_groups[0]['lr'] <= 1e-6:
                    # 计算开始训练时的总批次数
                    start_total_batches = 0
                    if self.checkpoint:
                        # 如果有检查点则从检查点获取代表本次训练的起始总批次数
                        start_total_batches = self.checkpoint['progress']['total_batches']

                    # 已训练的总批次数减去起始批次数代表本次训练的实际总批次数
                    # 如果本次训练的实际总批次数还没有达到预设的warmup比例，则代表处于warmup阶段
                    # 相反如果已经超过warmup阶段但是触发学习率过低，则说明达到目标轮次停止训练
                    if self.training_tracker.batch.total_batches - start_total_batches >= int(self.epoch_size.train.optimizer_steps * self.config.training.warmup_ratio):
                        self.logger.warning('Learning rate is too low, stopping training.')
                        if self.is_main_process: self.evaluate_epoch(quick_test=False)
                        # 退出训练
                        if self.is_distributed: distributed.destroy_process_group()
                        exit(0)

                if self.is_main_process and self.model_probe.active:
                    # 为了兼容梯度积累，非优化器步也要及时处理模型探针
                    self._log_model_probe_to_tb(prefix='Train/batch')
                    self.model_probe.set_active(active=False)

                if self.is_main_process and is_opt_step:
                    avg_loss = train_metrics_accumulator.avg_loss
                    avg_ppl = train_metrics_accumulator.avg_ppl

                    # 更新训练跟踪器
                    self.training_tracker.loss.train.step(loss=avg_loss)
                    self.training_tracker.ppl_meter.train.step(ppl=avg_ppl)

                    # 每 10 优化器步
                    if opt_step % 10 == 0:
                        # 记录到TensorBoard
                        self.summary_writer.add_scalars(main_tag='Train/batch/Loss', tag_scalar_dict={
                            'Value': avg_loss,
                            'Min': self.training_tracker.loss.train.min,
                            **{f'WindowAvg_{k}': v for k, v in self.training_tracker.loss.train.window_avg.items()}
                        }, global_step=self.training_tracker.batch.total_batches)
                        self.summary_writer.add_scalars(main_tag='Train/batch/PPL', tag_scalar_dict={
                            'Value': avg_ppl,
                            **{f'WindowAvg_{k}': v for k, v in self.training_tracker.ppl_meter.train.avg.items()}
                        }, global_step=self.training_tracker.batch.total_batches)
                        self.summary_writer.add_scalar(
                            tag='Train/batch/LearningRate', scalar_value=before_step_lr,
                            # step中已经执行过优化器进步了，所以这里记录的是step前的学习率，也就是本步实际执行的学习率
                            global_step=self.training_tracker.batch.total_batches
                        )

                    # 每 20 优化器步
                    if opt_step % 20 == 0:
                        pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'LR': f'{self.optimizer.param_groups[0]['lr']:.6f}'})
                        if self.config.training.gradient_clipping_max_norm > 0:
                            self.summary_writer.add_scalars(main_tag='Train/batch/GradClip', tag_scalar_dict={
                                'ClipRate': self.training_tracker.grad_clip_meter.clip_rate,  # 最近 100 优化器步的梯度裁剪应用率
                                'AvgClipScale': self.training_tracker.grad_clip_meter.avg_clip_scale,  # 最近 100 优化器步的平均裁剪比例
                            }, global_step=self.training_tracker.batch.total_batches)

                    # 每 50 批次
                    if opt_step % 50 == 0:
                        status_data = v_mem_status(device=self.device)
                        self.summary_writer.add_scalars(main_tag='Resource/VMEM', tag_scalar_dict={
                            'Allocated': status_data.allocated,
                            'Reserved': status_data.reserved,
                            'TotalMemory': status_data.device_total_memory,
                        }, global_step=self.training_tracker.batch.total_batches)

                    # 每 100 批次
                    if opt_step % 100 == 0:
                        # 计算并记录每秒训练批次数
                        current_time = time.time()
                        if speed_time is not None:
                            elapsed_time = current_time - speed_time
                            self.summary_writer.add_scalar(
                                tag='Train/batch/Speed_steps_per_second',
                                scalar_value=round(100 / elapsed_time, 2),
                                global_step=self.training_tracker.batch.total_batches
                            )
                        speed_time = current_time  # 重置计时器

                        self.summary_writer.add_scalars(
                            main_tag='Train/batch/grad_norm',
                            tag_scalar_dict=collect_layer_grad_norms(model=self.base_model),
                            global_step=self.training_tracker.batch.total_batches
                        )

                    # 每轮次验证 quick_eval_per_epoch 次
                    if self._buffer.eval_steps and opt_step in self._buffer.eval_steps:
                        pbar.disable = True  # 禁用 tqdm 输出
                        self.evaluate_epoch(quick_test=True)
                        self.model.train()  # 确保恢复训练模式
                        pbar.disable = False  # 启用 tqdm 输出
                        speed_time = None  # 验证会导致时间间隔不准确，重置计时器

        if self.is_main_process:
            # 记录到TensorBoard
            self.summary_writer.add_scalars(main_tag='Train/epoch/Loss', tag_scalar_dict={
                'Avg': self.training_tracker.loss.train.avg,
                'Max': self.training_tracker.loss.train.max,
                'Min': self.training_tracker.loss.train.min,
            }, global_step=self.training_tracker.epoch())

            # 评估模型
            self.evaluate_epoch(quick_test=False)

        # 轮次报告
        self.logger.info(f'Epoch {self.training_tracker.epoch()} completed. Average loss: {self.training_tracker.loss.train.avg:.4f}')

        # 更新训练守卫
        self.training_tracker.batch.reset()
        self.training_tracker.loss.train.reset()

    def start(self):
        """
        启动训练
        :return:
        """
        self.logger.info('Start training ...')
        while True:
            self.train_epoch()

            current_epoch = self.training_tracker.epoch()
            if self.config.training.num_epochs <= current_epoch:
                # 如果同一任务多个阶段可能会导致训练提前结束
                if self.checkpoint and self.TASK_TYPE == self.checkpoint['type'] and current_epoch - self.checkpoint['progress']['epoch'] < self.config.training.num_epochs:
                    continue
                if self.is_main_process and getattr(self, 'notifier', None) is not None:
                    self.notifier.notify(subject='Training Finished', content=f'Training has finished after {current_epoch} epochs.')
                self.logger.info('Training finished.')
                break


class PreTrainer(Trainer):
    """ Pre-trainer for QiChat Model """
    TASK_TYPE: str = 'PreTraining'

    def unpack_dataset(self, dataset: pandas.DataFrame) -> tuple[list[numpy.ndarray], list[numpy.ndarray]]:
        """Unpack dataset into inputs and labels"""
        inputs: list[numpy.ndarray] = dataset[self.config.dataset.field_name].tolist()
        labels: list[numpy.ndarray] = [ids.copy() for ids in inputs]

        return inputs, labels

    def init_loss_fn(self):
        """Initialize the loss function"""
        # 使用模型内部的 loss 函数
        self.logger.info(f'Using model\'s internal loss function.')

    def batch_preprocess(self, batch: list[tuple[torch.LongTensor, torch.LongTensor]]) -> tuple[BatchEncoding, None]:
        """
        Preprocess the batch
        :param batch:
        :return:
        """
        input_ids, label_ids = zip(*batch)

        input_ids = pad_sequence(sequences=input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        label_ids = pad_sequence(sequences=label_ids, batch_first=True, padding_value=-100)

        return BatchEncoding({
            'input_ids': input_ids,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id),  # 生成注意力掩码
            'labels': label_ids,
            'loss_ignore_index': -100,
        }), None

    @torch.no_grad()
    def inference_samples(self):
        """
        Inference some samples
        """
        self.model.eval()

        samples = [
            '量子纠缠是一种发生在多个粒子之间的量子关联现象，其特点是',
            '要在 Linux 上查看当前系统的内核版本，可以使用如下命令：',
            '明清时期的海上贸易在东亚格局中扮演了重要角色，其中',
            '通货膨胀通常指总体物价水平持续上涨的现象。在宏观经济分析中，',
            '雨停之后，巷子里仍残留着潮湿的气息。他站在屋檐下，',
            '“你觉得明天会下雨吗？”他抬头看了看天空，说：',
            'The French Revolution was a period of political and social upheaval in France that began',
            'A neural network learns by adjusting its parameters to minimize a loss function. In practice,',
            'During the Industrial Revolution, advances in manufacturing and transportation led to',
            'Inflation refers to a sustained increase in the general price level of goods and services. One common measure is',
            'Let X be a random variable with finite variance. Then the law of large numbers states that',
            'When the train finally arrived, she realized she had been waiting for hours, and'
        ]
        inputs = self.tokenizer(samples, return_tensors='pt', padding=True, padding_side='left', add_special_tokens=False)  # add_special_tokens=False 为去除 bos 和 eos
        outputs = self.base_model.generate(
            **inputs.to(self.device),
            generation_config=GenerationConfig(
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=[self.tokenizer.convert_tokens_to_ids('<|eot|>'), self.tokenizer.eos_token_id],
                use_cache=True,
                max_new_tokens=128,
                do_sample=False,  # 不进行采样（贪婪解码）
            )
        )
        outputs_trimmed: list[list[int]] = outputs[:, inputs.input_ids.size(1):].detach().to("cpu").tolist()  # 去掉 input 部分 [batch_size, gen_len]

        stop_set = {self.tokenizer.convert_tokens_to_ids("<|eot|>"), self.tokenizer.eos_token_id}
        # 在生成序列中找到第一个停止符位置并截断
        for i in range(len(outputs_trimmed)):
            for j in range(len(outputs_trimmed[i])):
                if outputs_trimmed[i][j] in stop_set:
                    outputs_trimmed[i] = outputs_trimmed[i][:j + 1]  # 保留停止符
                    break

        generated = self.tokenizer.batch_decode(outputs_trimmed, skip_special_tokens=False)
        for i, (input_text, generated_text) in enumerate(zip(samples, generated)):
            generated_text = str(generated_text).encode('utf-8', errors='replace').decode('utf-8')  # 确保生成文本中无法解码的字符被替换掉，避免 TensorBoard 显示错误
            self.summary_writer.add_text(
                tag=f'Inference/Samples_{i + 1}',
                text_string=f'Input: {input_text}\nGenerated: {generated_text}',
                global_step=self.training_tracker.batch.total_batches
            )


class FineTuner(PreTrainer):
    """ Fine-tuner for QiChat Model """
    TASK_TYPE: str = 'FineTuning'

    def unpack_dataset(self, dataset: pandas.DataFrame) -> tuple[list[numpy.ndarray], list[numpy.ndarray]]:
        """Unpack dataset into inputs and labels"""
        inputs: list[numpy.ndarray] = dataset[self.config.dataset.field_name].tolist()
        labels: list[numpy.ndarray] = []

        assistant_id = self.tokenizer.convert_tokens_to_ids('<|assistant|>')
        eot_id = self.tokenizer.convert_tokens_to_ids('<|eot|>')
        eos_id = self.tokenizer.eos_token_id

        for input_ids in inputs:
            label = input_ids.astype(numpy.int32, copy=True)

            # True = ignore(pad), False = keep(监督计算loss)
            ignore_mask = ~build_span_mask(
                input_ids=label,
                start_token_id=assistant_id,
                end_token_id=eot_id,
                include_start_token=False,
                include_end_token=True,
            )
            # eos也监督（SFT 阶段理论不应出现 eos ，而是使用 eot 作为段落结束标志）
            eos_pos = numpy.where(label == eos_id)[0]
            ignore_mask[eos_pos] = False

            label[ignore_mask] = -100  # 使用 -100 进行忽略，请确保与模型内部 loss 函数设置一致，能够正确忽略这些位置
            labels.append(label)

        return inputs, labels

    @torch.no_grad()
    def inference_samples(self):
        """
        Inference some samples
        """
        self.model.eval()

        # system_prompt = f'你是QiChat，现在是 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}。请根据用户提出的问题进行准确、简洁的回答，如果你不确定某个问题的答案，可以直接说“我不知道”。'
        samples = [
            '相对论的作者是谁？',
            'Who is the author of the theory of relativity?',
            '孔子是什么时期的人？',
            '美国共有多少个州？',
            'How many states are there in the United States?',
            '中国的全称是什么？',
            '什么是北回归线？',
            '32 加 45 等于多少？',
        ]
        input_chat_template = self.tokenizer.apply_chat_template(
            conversation=[
                [
                    # {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': sample}
                ] for sample in samples
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(input_chat_template, return_tensors='pt', padding=True, padding_side='left', add_special_tokens=False)

        # 截取每个样本的连续 [0, 218] token 以剔除 bos 和 换行符
        # 注意，tokenizer使用左填充，所以不可以直接 [:, 2:]
        cur, nxt = inputs.input_ids[:, :-1], inputs.input_ids[:, 1:]
        mask = (cur == 0) & (nxt == 218)
        inputs.input_ids[:, :-1][mask] = self.tokenizer.pad_token_id
        inputs.input_ids[:, 1:][mask] = self.tokenizer.pad_token_id
        inputs.attention_mask = inputs.input_ids != self.tokenizer.pad_token_id

        outputs = self.base_model.generate(
            **inputs.to(self.device),
            generation_config=GenerationConfig(
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=[self.tokenizer.convert_tokens_to_ids('<|eot|>'), self.tokenizer.eos_token_id],
                use_cache=True,
                max_new_tokens=128,
                do_sample=False
            ),
        )
        outputs_trimmed: list[list[int]] = outputs[:, inputs.input_ids.size(1):].detach().to("cpu").tolist()  # 去掉 input 部分 [batch_size, gen_len]

        stop_set = {self.tokenizer.convert_tokens_to_ids("<|eot|>"), self.tokenizer.eos_token_id}
        # 在生成序列中找到第一个停止符位置并截断
        for i in range(len(outputs_trimmed)):
            for j in range(len(outputs_trimmed[i])):
                if outputs_trimmed[i][j] in stop_set:
                    outputs_trimmed[i] = outputs_trimmed[i][:j + 1]  # 保留停止符
                    break

        generated = self.tokenizer.batch_decode(outputs_trimmed, skip_special_tokens=False)
        for i, (input_text, generated_text) in enumerate(zip(samples, generated)):
            generated_text = str(generated_text).encode('utf-8', errors='replace').decode('utf-8')
            self.summary_writer.add_text(
                tag=f'Inference/Samples_{i + 1}',
                text_string=f'Input: {input_text}\nGenerated: {generated_text}',
                global_step=self.training_tracker.batch.total_batches
            )


class DPOTrainer(Trainer):
    """ DPO trainer for QiChat Model """
    TASK_TYPE: str = 'DPOTraining'

    def __init__(self):
        raise NotImplementedError('DPOTrainer is not implemented yet.')


def ddp_worker(rank: int, nproc: int, trainer_class: type[Trainer]):
    """ DDP worker function """
    distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',
        world_size=nproc,
        rank=rank
    )
    torch.cuda.set_device(rank)

    try:
        trainer = trainer_class()
        trainer.start()
        distributed.destroy_process_group()
    except Exception as e:
        distributed.destroy_process_group()
        time.sleep(2.8)
        raise e
