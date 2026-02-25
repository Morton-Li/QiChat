from collections import deque
from typing import Optional

from .utils.type import DotDict


class Counter:
    """ Counter """
    def __init__(self, counter_name: str = 'counter'):
        self.counter_name: str = counter_name
        self._value: int = 0

    def __call__(self) -> int:
        """Get current counter"""
        return self._value

    @property
    def value(self) -> int:
        """Get current counter value"""
        return self._value

    def set(self, value: int) -> None:
        """
        Set counter
        Args:
            value (int): Value to set
        """
        self._value = int(value)

    def get(self) -> int:
        """Get current counter value"""
        return self._value

    def next(self) -> None:
        """Next counter"""
        self._value += 1

    def add(self, value: int) -> None:
        """
        Add value to counter
        Args:
            value (int): Value to add
        """
        self._value += int(value)

    def __add__(self, value: int) -> None:
        """
        Add value to counter using the + operator
        Args:
            value (int): Value to add
        """
        self.add(int(value))

    def reset(self) -> None:
        """Reset counter"""
        self._value = 0


class LossMeter:
    """ Loss Meter """
    def __init__(self, windows: Optional[list[int]] = None):
        self._max = float('-inf')
        self._min = float('inf')
        self._current = None
        self._total = 0.0
        # 最优平均值
        self._best_avg = float('inf')
        self._is_best = False
        self._step = Counter(counter_name='step')

        self.windows = windows
        if windows:
            self.values = deque(maxlen=max(windows))

    def step(self, loss: float) -> None:
        """
        Step
        Args:
            loss (float): Loss value
        """
        self._step.next()
        self._current = loss
        self._total += loss
        if self.windows:
            self.values.append(loss)
        if loss > self._max: self._max = loss
        if loss < self._min: self._min = loss

    def reset(self) -> None:
        """Reset"""
        self._max = float('-inf')
        self._min = float('inf')
        self._current = None
        self._total = 0.0
        self._step.reset()

    @property
    def avg(self) -> float:
        """Average"""
        if self._step.value == 0: return 0.0
        avg = self._total / self._step.value
        if avg < self._best_avg:
            self._best_avg = avg
            self._is_best = True
        return avg

    @property
    def window_avg(self) -> dict[str, float]:
        """Window Average"""
        if not self.windows or not self.values:
            return {}
        vals = list(self.values)
        n = len(vals)
        return {
            f'{(window / 1000) if (window % 1000 == 0) else round(window / 1000, 1)}k': round(sum(vals[-window:]) / min(n, window), 4)
            for window in self.windows
        }

    @property
    def max(self) -> float:
        """Max"""
        return self._max

    @property
    def min(self) -> float:
        """Min"""
        return self._min

    @property
    def is_best(self) -> bool:
        """Is best"""
        return self._is_best

    def reset_best(self) -> None:
        """Reset best flag"""
        self._is_best = False

    def reset_history(self) -> None:
        """Reset windows"""
        if self.windows:
            self.values.clear()


class PPLMeter:
    """ Perplexity Meter """
    def __init__(self, windows: list[int]):
        self.windows = windows
        self.values = deque(maxlen=max(windows))

    def step(self, ppl: float) -> None:
        """
        Step
        Args:
            ppl (float): Perplexity value
        """
        self.values.append(ppl)

    @property
    def avg(self) -> dict[str, float]:
        """Perplexity Average"""
        if not self.values:
            return {}
        vals = list(self.values)
        n = len(vals)
        return {
            f'{(window / 1000) if (window % 1000 == 0) else round(window / 1000, 1)}k': round(sum(vals[-window:]) / min(n, window), 4)
            for window in self.windows
        }

    def reset(self) -> None:
        """Reset"""
        self.values.clear()


class GradClipMeter:
    """ Gradient Clip Meter """
    def __init__(self, window: int = 100):
        self.is_clipped_record = deque(maxlen=window)
        self.clip_scale_record = deque(maxlen=window)

    def step(self, is_clipped: bool, clip_scale: float) -> None:
        """
        Step
        Args:
            is_clipped (bool): Whether the gradient is clipped
            clip_scale (float): The scale of the gradient clipping (e.g., clip_scale = 0.5 means the gradient is scaled to 50% of its original magnitude)
        """
        self.is_clipped_record.append(1.0 if is_clipped else 0.0)
        if is_clipped: self.clip_scale_record.append(clip_scale)
        else: self.clip_scale_record.append(0.0)

    @property
    def clip_rate(self) -> float:
        """Gradient Clip Rate"""
        if not self.is_clipped_record: return 0.0
        return round(sum(self.is_clipped_record) / len(self.is_clipped_record), 4)

    @property
    def avg_clip_scale(self) -> float:
        """Average Clip Scale (only for clipped steps)"""
        if not self.clip_scale_record: return 1.0
        clipped = sum(self.is_clipped_record)
        if clipped == 0: return 1.0  # 没发生裁剪，力度视为“无”（scale=1）
        return round(sum(self.clip_scale_record) / clipped, 4)


class BatchCounter(Counter):
    """ Batch Counter """
    def __init__(self):
        super().__init__(counter_name='batch')
        self._total_batches = 0

    @property
    def total_batches(self) -> int:
        """Get total batches"""
        return self._total_batches

    def set_total_batches(self, value: int) -> None:
        """
        Set counter
        Args:
            value (int): Value to set
        """
        self._total_batches = int(value)

    def get_total_batches(self) -> int:
        """Get current counter value"""
        return self._total_batches

    def next(self) -> None:
        """Next counter"""
        self._value += 1
        self._total_batches += 1

    def add(self, value: int) -> None:
        """
        Add value to counter
        Args:
            value (int): Value to add
        """
        self._value += int(value)
        self._total_batches += int(value)

    def reset_total_batches(self) -> None:
        """Reset total batches"""
        self._total_batches = 0


class TrainingTracker:
    def __init__(self):
        self.epoch = Counter(counter_name='epoch')
        self.batch = BatchCounter()
        self.loss = DotDict({
            'train': LossMeter(windows=[1000, 3000, 5000]),
            'valid': LossMeter(),
            'test': LossMeter()
        })
        self.ppl_meter = DotDict({
            'train': PPLMeter(windows=[1000, 3000, 5000]),
            'valid': PPLMeter(windows=[1000, 3000, 5000]),
            'test': PPLMeter(windows=[1000, 3000, 5000]),
        })
        self.grad_clip_meter = GradClipMeter(window=1000)


__all__ = ['TrainingTracker']
