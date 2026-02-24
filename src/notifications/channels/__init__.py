import importlib
import pkgutil

from .base import NotificationChannel


CHANNEL_REGISTRY: dict[str, type[NotificationChannel]] = {}


def _load_all_channels() -> None:
    for module in pkgutil.iter_modules(__path__):
        if module.name in ["base"]: continue

        lib = importlib.import_module(f'.{module.name}', package=__name__)
        for attr_name, attr in vars(lib).items():
            if isinstance(attr, type) and issubclass(attr, NotificationChannel) and attr is not NotificationChannel:
                CHANNEL_REGISTRY[attr_name] = attr


_load_all_channels()
