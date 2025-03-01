from typing import Any, Dict, Sequence, Tuple, Union, List
from dial_mpc.genesis.envs.unitree_h1_env import (
    UnitreeH1WalkEnvConfig,
    UnitreeH1WalkEnv,
)

_configs = {
    "unitree_h1_walk": UnitreeH1WalkEnvConfig,
}

_envs = {
    "unitree_h1_walk": UnitreeH1WalkEnv,
}


def register_config(name: str, config: Any):
    _configs[name] = config


def get_config(name: str) -> Any:
    return _configs[name]


def get_env(name: str) -> Any:
    return _envs[name]


def register_env(name: str, env: Any):
    _envs[name] = env
