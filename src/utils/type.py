from dataclasses import dataclass


class DotDict(dict):
    def __getattr__(self, item):
        value = self.get(item)
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[item] = value
        return value

    def __setattr__(self, key, value):
        # 若传入的 value 为普通 dict，则转换为 DotDict 以支持点语法访问
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        self[key] = value


@dataclass
class VMemStatus:
    allocated: float
    reserved: float
    device_total_memory: float
    reserved_ratio: float
    usage_ratio: float
