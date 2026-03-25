import datetime
import os
import sys
import threading
from typing import Optional, Literal

from .utils.path import log_path

# 日志级别
# DEBUG: 调试信息
# INFO: 一般信息
# WARNING: 警告
# ERROR: 错误
# CRITICAL: 严重错误
LogLevel = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

# 日志级别及其对应的名称和颜色
LOG_LEVELS_MAP = {
    1: {"name": "DEBUG", "color": "\033[0;36m"},    # Cyan
    2: {"name": "INFO", "color": "\033[0;32m"},     # Green
    3: {"name": "WARNING", "color": "\033[0;33m"},  # Yellow
    4: {"name": "ERROR", "color": "\033[0;31m"},    # Red
    5: {"name": "CRITICAL", "color": "\033[0;41m"}  # Background Red
}
RESET_COLOR = "\033[0m"


class Logger:
    def __init__(
        self,
        log_level: int | LogLevel = 2,  # 默认日志级别为 INFO
        log_filename: Optional[str] = None,
    ):
        self.log_level = 2  # 默认日志级别
        self.set_level(level=log_level)
        self.lock = threading.Lock()  # 线程锁
        self.log_filename = log_filename  # 日志文件名

        self._enabled = True  # 日志启用标志

    @staticmethod
    def get_log_level_number(log_level: str) -> int:
        """
        Convert a string log level to its corresponding numeric value.

        Args:
            log_level (str): The log level as a string (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').

        Returns:
            int: The numeric value of the log level.
        """
        # 查找日志级别对应的数字
        levels = LOG_LEVELS_MAP
        levels = {v['name']: k for k, v in levels.items()}  # Reverse the dictionary to map names to numbers
        if log_level not in levels:
            raise ValueError(f"Invalid log level: {log_level}. Valid levels are: {', '.join(levels.keys())}")
        return levels.get(log_level.upper())  # Default to INFO level if not found

    def set_level(self, level: int | LogLevel) -> 'Logger':
        if isinstance(level, str):
            level = self.get_log_level_number(level)
        if level not in LOG_LEVELS_MAP:
            raise ValueError(f"Invalid log level: {level}")
        self.log_level = level
        return self

    def log(self, message: str, level: int, newline: bool = True):
        if not self.is_enabled: return
        if level not in LOG_LEVELS_MAP:
            raise ValueError(f"Invalid log level: {level}")
        # 如果日志级别低于指定的日志级别，则不输出
        if level < self.log_level:
            return

        # 选择的日志级别
        log_level_selected = LOG_LEVELS_MAP[level]

        # 构造日志消息
        log_message = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{log_level_selected['name']}] {message}"

        with self.lock:
            # 如果指定了文件名，并且格式为str，则将日志写入文件
            if self.log_filename and isinstance(self.log_filename, str):
                if not os.path.exists(log_path()):
                    os.makedirs(log_path())
                with open(log_path(self.log_filename), 'a') as f:
                    f.write(log_message + '\n') if newline else f.write(log_message)

            if sys.stdout.isatty(): log_message = f"{log_level_selected["color"]}{log_message}{RESET_COLOR}"
            print(log_message) if newline else print(log_message, end='', flush=True)

    def debug(self, message: str, newline: bool = True): self.log(message, 1, newline)
    def info(self, message: str, newline: bool = True): self.log(message, 2, newline)
    def warning(self, message: str, newline: bool = True): self.log(message, 3, newline)
    def error(self, message: str, newline: bool = True): self.log(message, 4, newline)
    def critical(self, message: str, newline: bool = True): self.log(message, 5, newline)

    def append(self, message: str, level: int | str = 'INFO', newline: bool = True):
        """
        追加日志消息到当前日志文件。
        :param message: 日志消息
        :param level: 日志级别，默认为 INFO
        :param newline: 是否在日志消息后添加换行符，默认为 True
        """
        if not self.is_enabled: return
        level = self.get_log_level_number(level) if isinstance(level, str) else level
        if level not in LOG_LEVELS_MAP:
            raise ValueError(f"Invalid log level: {level}")
        # 如果日志级别低于指定的日志级别，则不输出
        if level < self.log_level:
            return

        log_level_selected = LOG_LEVELS_MAP[level]

        with self.lock:
            # 如果指定了文件名，并且格式为str，则将日志写入文件
            if self.log_filename and isinstance(self.log_filename, str):
                if not os.path.exists(log_path()):
                    os.makedirs(log_path())
                with open(log_path(self.log_filename), 'a') as f:
                    f.write(message + '\n') if newline else f.write(message)

            if sys.stdout.isatty(): message = f"{log_level_selected["color"]}{message}{RESET_COLOR}"
            print(message) if newline else print(message, end='', flush=True)

    def enable(self): self._enabled = True
    def disable(self): self._enabled = False
    @property
    def is_enabled(self) -> bool: return self._enabled


_logger_instance = None
_logger_lock = threading.Lock()

def get_logger(log_filename: Optional[str] = "app.log", log_level: int | str = 'INFO') -> Logger:
    global _logger_instance
    with _logger_lock:
        if _logger_instance is None:
            _logger_instance = Logger(log_filename=log_filename, log_level=log_level)
        return _logger_instance


__all__ = ["Logger", "get_logger", "LogLevel"]
