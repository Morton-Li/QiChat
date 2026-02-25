from .channels import CHANNEL_REGISTRY
from .channels.base import NotificationChannel


class Notifier:
    def __init__(self, channels: list[NotificationChannel]) -> None:
        self._channels = channels

    @classmethod
    def from_config(cls, config: dict[str, dict[str, str]]) -> "Notifier":
        # 检查 CHANNEL_REGISTRY.keys 是否包含所有 config_channels
        config_channels = config.keys()
        unsupported_channels = [
            channel
            for channel in config.keys()
            if channel not in CHANNEL_REGISTRY
        ]
        if unsupported_channels:
            raise ValueError(f"Notification channels '{', '.join(unsupported_channels)}' are not supported.")

        return cls(channels=[
            CHANNEL_REGISTRY[channel](**config[channel])
            for channel in config_channels
        ])

    def notify(self, subject: str, content: str) -> None:
        for ch in self._channels: ch.send(subject=subject, content=content)
