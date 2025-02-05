from typing import Any, Dict, Sequence, Tuple, Union, List
from dial_mpc.envs.unitree_h1_env import (
    UnitreeH1WalkEnvConfig,
    UnitreeH1PushCrateEnvConfig,
    UnitreeH1LocoEnvConfig,
    UnitreeH1WalkEnv,
    UnitreeH1PushCrateEnv,
    UnitreeH1LocoEnv,
)
from dial_mpc.envs.unitree_go2_env import (
    UnitreeGo2EnvConfig,
    UnitreeGo2SeqJumpEnvConfig,
    UnitreeGo2CrateEnvConfig,
    UnitreeGo2Env,
    UnitreeGo2SeqJumpEnv,
    UnitreeGo2CrateEnv,
)

_configs = {
    "unitree_h1_walk": UnitreeH1WalkEnvConfig,
    "unitree_h1_push_crate": UnitreeH1PushCrateEnvConfig,
    "unitree_h1_loco": UnitreeH1LocoEnvConfig,
    "unitree_go2_walk": UnitreeGo2EnvConfig,
    "unitree_go2_seq_jump": UnitreeGo2SeqJumpEnvConfig,
    "unitree_go2_crate_climb": UnitreeGo2CrateEnvConfig,
}

_envs = {
    "unitree_h1_walk": UnitreeH1WalkEnv,
    "unitree_h1_push_crate": UnitreeH1PushCrateEnv,
    "unitree_h1_loco": UnitreeH1LocoEnv,
    "unitree_go2_walk": UnitreeGo2Env,
    "unitree_go2_seq_jump": UnitreeGo2SeqJumpEnv,
    "unitree_go2_crate_climb": UnitreeGo2CrateEnv,
}


def register_config(name: str, config: Any):
    _configs[name] = config


def get_config(name: str) -> Any:
    return _configs[name]


def get_environment(name: str) -> Any:
    return _envs[name]


def register_environment(name: str, env: Any):
    _envs[name] = env
