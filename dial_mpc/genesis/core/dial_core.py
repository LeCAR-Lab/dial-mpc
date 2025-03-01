import os
import time
from dataclasses import dataclass
import importlib
import sys
import copy

import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import art
import emoji
import math
import torch
import torch.nn.functional as F
import genesis as gs

import dial_mpc.genesis.envs as dial_envs
from dial_mpc.genesis.utils.io_utils import load_dataclass_from_dict, get_example_path
from dial_mpc.genesis.examples import examples
from dial_mpc.core.dial_config import DialConfig

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn


def rollout_us(step_env, us):
    rewards = torch.zeros(us.shape[0], us.shape[1], device=gs.device)
    for i, u in enumerate(us):
        obs, rew = step_env(u)
        rewards[i] = rew
    return rewards


def softmax_update(weights, Y0s, sigma, mu_0t):
    mu_0tm1 = torch.einsum("n,nij->ij", weights, Y0s)
    return mu_0tm1, sigma


class MBDPI:
    def __init__(self, args: DialConfig, env):
        self.args = args
        self.env = env
        self.nu = env.action_size

        self.update_fn = {
            "mppi": softmax_update,
        }[args.update_method]

        sigma0 = 1e-2
        sigma1 = 1.0
        A = sigma0
        B = math.log(sigma1 / sigma0) / args.Ndiffuse
        self.sigmas = A * torch.exp(B * torch.arange(args.Ndiffuse))
        self.sigma_control = (
            args.horizon_diffuse_factor ** torch.arange(args.Hnode + 1)[::-1]
        )

        # node to u
        self.ctrl_dt = 0.02
        self.step_us = torch.linspace(0, self.ctrl_dt * args.Hsample, args.Hsample + 1)
        self.step_nodes = torch.linspace(0, self.ctrl_dt * args.Hsample, args.Hnode + 1)
        self.node_dt = self.ctrl_dt * (args.Hsample) / (args.Hnode)

        # setup function
        self.rollout_us = lambda us: rollout_us(self.env.step, us)

    def node2u(self, nodes):
        """
        nodes: (Hnode, Nsample, nu)
        Returns: (Hsample, Nsample, nu)
        """
        return F.interpolate(nodes, mode="linear", scale_factor=self.args.Hsample / self.args.Hnode)

    def u2node(self, us):
        """
        us: (Hsample, Nsample, nu)
        Returns: (Hnode, Nsample, nu)
        """
        return F.interpolate(us, mode="linear", scale_factor=self.args.Hnode / self.args.Hsample)

    def reverse_once(self, Ybar_i, noise_scale):
        # sample from q_i
        eps_Y = torch.randn(
            (self.args.Nsample, self.args.Hnode + 1, self.nu), device=gs.device
        )
        Y0s = eps_Y * noise_scale[None, :, None] + Ybar_i
        # we can't change the first control
        Y0s = Y0s.at[:, 0].set(Ybar_i[0, :])
        # append Y0s with Ybar_i to also evaluate Ybar_i
        Y0s = torch.cat([Y0s, Ybar_i[None]], dim=0)
        Y0s = torch.clip(Y0s, -1.0, 1.0)
        # convert Y0s to us
        us = self.node2u(Y0s)

        # esitimate mu_0tm1
        rewss = self.rollout_us(us)
        rew_Ybar_i = rewss[-1].mean()

        rews = rewss.mean(axis=-1)
        logp0 = (rews - rew_Ybar_i) / rews.std(axis=-1) / self.args.temp_sample

        weights = F.softmax(logp0, dim=-1)
        Ybar, new_noise_scale = self.update_fn(weights, Y0s, noise_scale, Ybar_i)

        # NOTE: update only with reward
        Ybar = torch.einsum("n,nij->ij", weights, Y0s)

        return Ybar, rews

    def shift(self, Y):
        u = self.node2u(Y)
        u = torch.roll(u, -1, dims=0)
        u = u.at[-1].set(torch.zeros(self.nu))
        Y = self.u2node(u)
        return Y


def main():

    # initialize genesis
    gs.init(backend=gs.gpu)

    def reverse_scan(Y0, factor):
        Y0, rews = mbdpi.reverse_once(Y0, factor)
        return Y0, rews

    art.tprint("LeCAR @ CMU\nDIAL-MPC", font="big", chr_ignore=True)
    parser = argparse.ArgumentParser()
    config_or_example = parser.add_mutually_exclusive_group(required=True)
    config_or_example.add_argument("--config", type=str, default=None)
    config_or_example.add_argument("--example", type=str, default=None)
    config_or_example.add_argument("--list-examples", action="store_true")
    parser.add_argument(
        "--custom-env",
        type=str,
        default=None,
        help="Custom environment to import dynamically",
    )
    args = parser.parse_args()

    if args.list_examples:
        print("Examples:")
        for example in examples:
            print(f"  {example}")
        return

    if args.custom_env is not None:
        sys.path.append(os.getcwd())
        importlib.import_module(args.custom_env)

    if args.example is not None:
        config_dict = yaml.safe_load(open(get_example_path(args.example + ".yaml")))
    else:
        config_dict = yaml.safe_load(open(args.config))

    dial_config = load_dataclass_from_dict(DialConfig, config_dict)

    # set seed
    torch.manual_seed(dial_config.seed)
    torch.cuda.manual_seed(dial_config.seed)

    # find env config
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config_dict, convert_list_to_array=True
    )

    print(emoji.emojize(":rocket:") + " Creating environment")
    parallel_envs = dial_envs.get_env(dial_config.env_name)(env_config, dial_config.Nsample)
    main_env = dial_envs.get_env(dial_config.env_name)(env_config, 1, render=True)
    mbdpi = MBDPI(dial_config, parallel_envs)

    YN = torch.zeros([dial_config.Hnode + 1, mbdpi.nu])

    # Y0 = mbdpi.reverse(state_init, YN, rng_exp)
    Y0 = YN

    Nstep = dial_config.n_steps
    rews = []
    us = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        TextColumn("[bold green]Rew: {task.fields[rew]}"),
        TextColumn("[bold blue]Freq: {task.fields[freq]} Hz"),
        expand=True,
    ) as progress:
        task = progress.add_task(
            "Rollout",
            total=Nstep,
            rew="0.00e+00",
            freq="0.00",
        )
        for t in range(Nstep):
            # forward single step
            obs, rew = main_env.step(Y0[0])
            rews.append(rew)
            us.append(Y0[0])

            # update Y0
            Y0 = mbdpi.shift(Y0)

            n_diffuse = dial_config.Ndiffuse
            if t == 0:
                n_diffuse = dial_config.Ndiffuse_init
                print("Performing JIT on DIAL-MPC")

            t0 = time.time()
            traj_diffuse_factors = (
                mbdpi.sigma_control * dial_config.traj_diffuse_factor ** (torch.arange(n_diffuse))[:, None]
            )
            for i in range(len(traj_diffuse_factors)):
                mbdpi.env.reset(main_env)
                mbdpi.env.pre_step()
                Y0, rews = reverse_scan(
                    Y0, traj_diffuse_factors[i]
                )
                mbdpi.env.post_step()
            freq = 1 / (time.time() - t0)
            progress.update(
                task,
                advance=1,
                rew=f"{rew.item():.2e}",
                freq=f"{freq:.2f}",
            )


if __name__ == "__main__":
    main()
