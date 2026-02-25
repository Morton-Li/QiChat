import json
import os
from pathlib import Path

from .utils.type import DotDict
from .utils.path import root_path


DEFAULT_CONFIG: dict[str, int|float|bool|str] = {
    "device": "auto",
    "log_level": "INFO",
    "model": {
        "param_size": "0.3B",  # 参数量
        "attn_implementation": "flash_attention_2",
        "dtype": 'bfloat16',
        "enable_grad_checkpointing": False,
        "model_config_kwargs": {},
    },
    "training": {
        "batch_size": 16,  # 每个设备的批次大小
        "learning_rate": 2.2e-4,
        "min_learning_rate": 2.2e-5,
        "num_epochs": 1,
        "warmup_ratio": 0.018,
        "gradient_clipping_max_norm": 1.2,
        "quick_eval_per_epoch": 35,  # 在每个 epoch 中进行多少次快速评估
        "freeze_layers": 0,  # 冻结前 n 层参数不进行训练
        "freeze_embeddings": False,
        "grad_accum_steps": 1,  # 梯度累积步数
    },
    "seed": 36,
    "dataset": {
        "max_input_length": -1,
        "test_split": 0.05,
        "field_name": "tokenized_ids",
    },
    "notifications": {},
    "max_checkpoint_count": 18,
}
CONFIG_PATH: Path = root_path('config.json')


def check_config_integrity(config: dict, correct_config: dict) -> bool | list[str]:
    """ Check config integrity """
    missing = []
    for key, value in correct_config.items():
        if key not in config:
            missing.append(key)
            continue
        if isinstance(value, dict):
            if not isinstance(config[key], dict):
                missing.append(key)  # Type mismatch
            else:
                check_result = check_config_integrity(config[key], correct_config[key])
                if check_result is not True:
                    for child_key in check_result:
                        missing.append(f'{key}.{child_key}')
    return True if not missing else missing


def create_default_config() -> None:
    """ Create default config file """
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=4)


def get_config() -> DotDict:
    """ Get config instance """
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        missing_check = check_config_integrity(config=config, correct_config=DEFAULT_CONFIG)
        if missing_check is not True:
            print(f'Config file is missing the following fields: {[item for item in missing_check]}')
            exit(0)
        return DotDict(config)
    else:
        create_default_config()
        print(f'Config file not found, default config file created at {CONFIG_PATH}, please modify the config file according to the actual situation.')
        exit(0)
