import multiprocessing
from typing import Literal

import psutil
import torch

from .type import VMemStatus


def is_fork_supported() -> bool:
    """
    Check if the current process is a fork process.
    :return:
    """
    try:
        return multiprocessing.get_start_method(allow_none=True) == 'fork'
    except RuntimeError:
        return False


def v_mem_status(device: Literal['cpu', 'cuda', 'mps'] | torch.device) -> VMemStatus:
    """
    检查Video内存状态
    Args:
        device(Literal['cpu', 'cuda', 'mps'] | torch.device): 设备类型
    Returns:
        VMemStatus: 包含内存使用情况的字典
    """
    device_type = device.type if isinstance(device, torch.device) else device
    if device_type == 'cuda':
        device_total_memory = torch.cuda.get_device_properties(device).total_memory  # GPU总内存
        allocated = torch.cuda.memory_allocated(device=device)  # 当前已分配的内存，实际上使用的内存
        reserved = torch.cuda.memory_reserved(device=device)  # 当前已缓存的内存，虽然没有使用但是已经申请的将来可能会使用的内存

        return VMemStatus(
            allocated=allocated / (1024 ** 2),
            reserved=reserved / (1024 ** 2),
            device_total_memory=device_total_memory / (1024 ** 2),
            reserved_ratio=round(reserved / device_total_memory, 6),
            usage_ratio=round(allocated / device_total_memory, 6),
        )

    elif device_type == 'mps':
        total_memory = torch.mps.driver_allocated_memory()  # MPS总内存
        allocated = torch.mps.current_allocated_memory()  # 当前已分配的内存

        return VMemStatus(
            allocated=allocated / (1024 ** 2),
            reserved=-1,
            device_total_memory=total_memory / (1024 ** 2),
            reserved_ratio=-1,
            usage_ratio=allocated / total_memory,
        )

    elif device_type == 'cpu':
        # 获取RAM
        memory = psutil.virtual_memory()
        total_memory = memory.total
        available = memory.available

        return VMemStatus(
            allocated=memory.used / (1024 ** 2),  # 已分配
            reserved=(total_memory - available) / (1024 ** 2),
            device_total_memory=total_memory / (1024 ** 2),
            reserved_ratio=round((total_memory - available) / total_memory, 6),
            usage_ratio=memory.percent / 100,
        )

    else:
        # 不支持的设备类型
        return VMemStatus(
            allocated=-1,
            reserved=-1,
            device_total_memory=-1,
            reserved_ratio=-1,
            usage_ratio=-1,
        )
