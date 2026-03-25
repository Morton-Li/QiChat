from pathlib import Path

from src.utils.path import data_path


def format_count(value: int) -> str:
    if value < 1e3: return str(value)
    if value < 1e6: return f'{value / 1e3:.2f}K'
    if value < 1e9: return f'{value / 1e6:.2f}M'
    return f'{value / 1e9:.2f}B'


def rel_to_data_dir(target_path: Path) -> Path: return target_path.relative_to(data_path())
