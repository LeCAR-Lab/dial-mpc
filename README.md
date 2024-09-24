# DIAL-MPC: Diffusion-Inspired Annealing For Legged MPC

<div align="center">

[[Website]](https://lecar-lab.github.io/dial-mpc/)
[[PDF]](https://drive.google.com/file/d/1Z39MCvnl-Tdraon4xAj37iQYLsUh5UOV/view?usp=sharing)
[[Arxiv(Coming Soon)]](https://arxiv.org/)

[<img src="https://img.shields.io/badge/Backend-Jax-red.svg"/>](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<img src="assets/joint.gif" width="600px"/>

</div>

This repository contains the code (simulation and real-world experiments with minimum setup) for the paper "Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing".

DIAL-MPC is a novel sampling-based MPC framework for legged robot ***full-order torque-level*** control with both precision and agility in a ***training-free*** manner. 
DIAL-MPC is designed to be simple and flexible, with minimal requirements for specific reward design and dynamics model.
That means you can test out the controller in a plug-and-play manner with minimum setup.

## Setup

### Install `unitree_sdk2_python`
Execute the following commands in the terminal:

```bash
cd ~
sudo apt install python3-pip
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```

### Install `dial-mpc`

```bash
git clone https://github.com/LeCar-Lab/dial-mpc.git --depth 1
cd dial-mpc
pip3 install -e .
```

## Usage

### Simulation

#### Run Examples

List available examples:

```bash
dial-mpc --list-examples
```

Run an example:

```bash
dial-mpc --example unitree_h1_jog
```

After rollout completes, go to `127.0.0.1:5000` to visualize the rollouts.

#### Contributing New Environment

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
### Real-World Experiments

#### Sim2sim test

We provide a asynchronous simulation environment for driver-in-the-loop testing with dynamic mismatch and control latency. It is recommended to test the controller with the real robot after the sim2sim test.

#### Sim2real test

### Rendering the results

If you want better visualization, you can check out the `render` branch for the blender visualization examples. 

## Acknowledgements

* This codebase's environment and RL implementation is built on top of [Brax](https://github.com/google/brax).
* We use [Mujoco MJX](https://github.com/deepmind/mujoco) for the physics engine.
* Controller design and implementation is inspired by [Model-based Diffusion](https://github.com/LeCAR-Lab/model-based-diffusion)


## BibTeX

If you find this code useful for your research, please consider citing:

```bibtex

```