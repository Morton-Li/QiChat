from pathlib import Path


def root_path(*args) -> Path:
    """
    Get the root directory of the project
    :param args:
    :return:
    """
    root = Path(__file__).parent.parent.parent.absolute()
    return root.joinpath(*args)

def src_path(*args) -> Path:
    """
    Get the src directory
    :param args:
    :return:
    """
    return root_path('src', *args)

def data_path(*args) -> Path:
    """
    Get the data directory
    :param args:
    :return:
    """
    return root_path('data', *args)

def log_path(*args) -> Path:
    """
    Get the log directory
    :param args:
    :return:
    """
    return root_path('logs', *args)
