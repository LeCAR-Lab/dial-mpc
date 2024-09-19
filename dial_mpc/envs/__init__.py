from typing import Any, Dict, Sequence, Tuple, Union, List

_configs = {}


def register_config(name: str, config: Any):
    _configs[name] = config


def get_config(name: str) -> Any:
    return _configs[name]
