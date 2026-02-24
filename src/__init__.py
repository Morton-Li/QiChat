from src.utils.path import root_path

# 确保所需目录已被创建
for path in [
    'data',
    'data/checkpoint',
    'data/dataset',
    'data/tokenizer',
    'data/tokenizer/model',
]: root_path(path).mkdir(parents=True, exist_ok=True)
