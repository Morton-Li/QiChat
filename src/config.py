import json
import os
from pathlib import Path

from .utils.type import TrainerConfig
from .utils.path import root_path


CONFIG_PATH: Path = root_path('config.json')


def create_default_config() -> None:
    """ Create default config file """
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(TrainerConfig().to_dict(), f, ensure_ascii=False, indent=4)


def get_config() -> TrainerConfig:
    """ Get config instance """
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return TrainerConfig.from_dict(config)
    else:
        create_default_config()
        print(f'Config file not found, default config file created at {CONFIG_PATH.relative_to(root_path())}, please modify the config file according to the actual situation.')
        exit(0)
