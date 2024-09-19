# dial-mpc

## Install `unitree_sdk2_python`
Execute the following commands in the terminal:

```bash
cd ~
sudo apt install python3-pip
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```

## Run Examples

List available examples:

```bash
dial-mpc --list-examples
```

Run an example:

```bash
dial-mpc --example unitree_h1_jog
```

After rollout completes, go to `127.0.0.1:5000` to visualize the rollouts.

## Contributing New Environment

1. Import the base environment and config
2. Implement required functions
3. Register environment
4. Configure config file
5. Run with `dial-mpc --config <config.yaml>`

Example environment:

```python
from dataclasses import dataclass

from brax import envs as brax_envs
from brax.envs.base import State

from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
import dial_mpc.envs as dial_envs

@dataclass
class MyEnvConfig(BaseEnvConfig):
    arg1: 1.0
    arg2: "test"

class MyEnv(BaseEnv):
    def __init__(self, config: MyEnvConfig):
        super().__init__(config)
        # custom initializations below...

    def make_system(self, config: MyEnvConfig) -> System:
        model_path = ("my_model/my_model.xml")
        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def reset(self, rng: jax.Array) -> State:
        # TODO: implement reset

    def step(self, state: State, action: jax.Array) -> State:
        # TODO: implement step

brax_envs.register_environment("my_env_name", MyEnv)
dial_envs.register_config("my_env_name", MyEnvConfig)
```

Example configuration file:
```yaml
# DIAL-MPC
seed: 0
output_dir: dial_mpc_ws/my_model
n_steps: 400

env_name: my_env_name
Nsample: 2048
Hsample: 25
Hnode: 5
Ndiffuse: 4
Ndiffuse_init: 10
temp_sample: 0.05
horizon_diffuse_factor: 1.0
traj_diffuse_factor: 0.5
update_method: mppi


# Base environment
dt: 0.02
timestep: 0.02
leg_control: torque
action_scale: 1.0

# My Env
arg1: 2.0
arg2: "test_2"
```