import os
import time
from dataclasses import dataclass
from typing import Callable
import importlib
import sys

import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots
import art
import emoji

import jax
from jax import numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import functools
from jax.tree_util import Partial

import mujoco
import mujoco.mjx as mjx
from mujoco.mjx import Data

from brax.io import html
from brax.base import State as PipelineState
from brax.base import Transform, Motion, Contact

import dial_mpc.envs as dial_envs
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
from dial_mpc.examples import examples
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.envs.base_env import DialState, BaseEnv
plt.style.use("science")

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags


def rollout_us(step_env: Callable[[DialState, jax.Array], DialState], state: DialState, us: jax.Array):
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.data)

    _, (rews, pipline_states) = jax.lax.scan(step, state, us)
    return rews, pipline_states


@jax.jit
def softmax_update(weights, Y0s, sigma, mu_0t):
    mu_0tm1 = jnp.einsum("n,nij->ij", weights, Y0s)
    return mu_0tm1, sigma


class MBDPI:
    def __init__(self, args: DialConfig, env: BaseEnv):
        self.args = args
        self.env = env
        self.nu = env.nu

        self.update_fn = {
            "mppi": softmax_update,
        }[args.update_method]

        sigma0 = 1e-2
        sigma1 = 1.0
        A = sigma0
        B = jnp.log(sigma1 / sigma0) / args.Ndiffuse
        self.sigmas = A * jnp.exp(B * jnp.arange(args.Ndiffuse))
        self.sigma_control = (
            args.horizon_diffuse_factor ** jnp.arange(args.Hnode + 1)[::-1]
        )

        # node to u
        self.ctrl_dt = 0.02
        self.step_us = jnp.linspace(0, self.ctrl_dt * args.Hsample, args.Hsample + 1)
        self.step_nodes = jnp.linspace(0, self.ctrl_dt * args.Hsample, args.Hnode + 1)
        self.node_dt = self.ctrl_dt * (args.Hsample) / (args.Hnode)

        # setup function
        self.rollout_us = jax.jit(functools.partial(rollout_us, self.env.step))
        self.rollout_us_vmap = jax.jit(jax.vmap(self.rollout_us, in_axes=(None, 0)))
        self.node2u_vmap = jax.jit(
            jax.vmap(self.node2u, in_axes=(1), out_axes=(1))
        )  # process (horizon, node)
        self.u2node_vmap = jax.jit(jax.vmap(self.u2node, in_axes=(1), out_axes=(1)))
        self.node2u_vvmap = jax.jit(
            jax.vmap(self.node2u_vmap, in_axes=(0))
        )  # process (batch, horizon, node)
        self.u2node_vvmap = jax.jit(jax.vmap(self.u2node_vmap, in_axes=(0)))

    @functools.partial(jax.jit, static_argnums=(0,))
    def node2u(self, nodes):
        spline = InterpolatedUnivariateSpline(self.step_nodes, nodes, k=2)
        us = spline(self.step_us)
        return us

    @functools.partial(jax.jit, static_argnums=(0,))
    def u2node(self, us):
        spline = InterpolatedUnivariateSpline(self.step_us, us, k=2)
        nodes = spline(self.step_nodes)
        return nodes

    @functools.partial(jax.jit, static_argnums=(0,))
    def reverse_once(self, state, rng, Ybar_i, noise_scale):
        # sample from q_i
        rng, Y0s_rng = jax.random.split(rng)
        eps_Y = jax.random.normal(
            Y0s_rng, (self.args.Nsample, self.args.Hnode + 1, self.nu)
        )
        Y0s = eps_Y * noise_scale[None, :, None] + Ybar_i
        # we can't change the first control
        Y0s = Y0s.at[:, 0].set(Ybar_i[0, :])
        # append Y0s with Ybar_i to also evaluate Ybar_i
        Y0s = jnp.concatenate([Y0s, Ybar_i[None]], axis=0)
        Y0s = jnp.clip(Y0s, -1.0, 1.0)
        # convert Y0s to us
        us = self.node2u_vvmap(Y0s)

        # esitimate mu_0tm1
        rewss, mjx_datass = self.rollout_us_vmap(state, us)
        rew_Ybar_i = rewss[-1].mean()
        qss = mjx_datass.qpos
        qdss = mjx_datass.qvel
        xss = mjx_datass.xpos
        rews = rewss.mean(axis=-1)
        logp0 = (rews - rew_Ybar_i) / rews.std(axis=-1) / self.args.temp_sample

        weights = jax.nn.softmax(logp0)
        Ybar, new_noise_scale = self.update_fn(weights, Y0s, noise_scale, Ybar_i)

        # NOTE: update only with reward
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)
        qbar = jnp.einsum("n,nij->ij", weights, qss)
        qdbar = jnp.einsum("n,nij->ij", weights, qdss)
        xbar = jnp.einsum("n,nijk->ijk", weights, xss)

        info = {
            "rews": rews,
            "qbar": qbar,
            "qdbar": qdbar,
            "xbar": xbar,
            "new_noise_scale": new_noise_scale,
        }

        return rng, Ybar, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def shift(self, Y):
        u = self.node2u_vmap(Y)
        u = jnp.roll(u, -1, axis=0)
        u = u.at[-1].set(jnp.zeros(self.nu))
        Y = self.u2node_vmap(u)
        return Y

    def shift_Y_from_u(self, u, n_step):
        u = jnp.roll(u, -n_step, axis=0)
        u = u.at[-n_step:].set(jnp.zeros_like(u[-n_step:]))
        Y = self.u2node_vmap(u)
        return Y


def main():

    def reverse_scan(rng_Y0_state, factor):
        rng, Y0, state = rng_Y0_state
        rng, Y0, info = mbdpi.reverse_once(state, rng, Y0, factor)
        return (rng, Y0, state), info

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
    rng = jax.random.PRNGKey(seed=dial_config.seed)

    # find env config
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config_dict, convert_list_to_array=True
    )

    print(emoji.emojize(":rocket:") + " Creating environment")
    env: BaseEnv = dial_envs.get_environment(dial_config.env_name)(env_config)
    step_env = jax.jit(env.step)
    mbdpi = MBDPI(dial_config, env)

    rng, rng_reset = jax.random.split(rng)
    state_init = env.reset(rng_reset)

    YN = jnp.zeros([dial_config.Hnode + 1, mbdpi.nu])

    rng_exp, rng = jax.random.split(rng)
    Y0 = YN

    Nstep = dial_config.n_steps
    rews = []
    rews_plan = []
    rollout = []
    state = state_init
    us = []
    infos = []
    with tqdm(range(Nstep), desc="Rollout") as pbar:
        for t in pbar:
            # forward single step
            state = step_env(state, Y0[0])
            rollout.append(state.data)
            rews.append(state.reward)
            us.append(Y0[0])
            
            # update Y0
            Y0 = mbdpi.shift(Y0)

            n_diffuse = dial_config.Ndiffuse
            if t == 0:
                n_diffuse = dial_config.Ndiffuse_init
                print("Performing JIT on DIAL-MPC")

            t0 = time.time()
            traj_diffuse_factors = (
                mbdpi.sigma_control * dial_config.traj_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]
            )
            (rng, Y0, _), info = jax.lax.scan(
                reverse_scan, (rng, Y0, state), traj_diffuse_factors
            )
            rews_plan.append(info["rews"][-1].mean())
            infos.append(info)
            freq = 1 / (time.time() - t0)
            pbar.set_postfix({"rew": f"{state.reward:.2e}", "freq": f"{freq:.2f}"})

    rew = jnp.array(rews).mean()
    print(f"mean reward = {rew:.2e}")

    # save us
    # us = jnp.array(us)
    # jnp.save("./results/us.npy", us)

    # create result dir if not exist
    if not os.path.exists(dial_config.output_dir):
        os.makedirs(dial_config.output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # plot rews_plan
    # plt.plot(rews_plan)
    # plt.savefig(os.path.join(dial_config.output_dir,
    #             f"{timestamp}_rews_plan.pdf"))

    # host webpage with flask
    print("Processing rollout for visualization")

    # save the rollout
    data = []
    xdata = []
    brax_rollout = []
    for i in range(len(rollout)):
        mjx_data: Data = rollout[i]
        data.append(
            jnp.concatenate(
                [
                    jnp.array([i]),
                    mjx_data.qpos,
                    mjx_data.qvel,
                    mjx_data.ctrl,
                ]
            )
        )
        xdata.append(infos[i]["xbar"][-1])
        brax_state = PipelineState(
            q=mjx_data.qpos,
            qd=mjx_data.qvel,
            x=Transform(pos=mjx_data.xpos, rot=mjx_data.xquat),
            xd=Motion(vel=mjx_data.cvel[:, 3:], ang=mjx_data.cvel[:, :3] * 180 / jnp.pi),
            contact=Contact(
                dist=mjx_data.contact.dist,
                pos=mjx_data.contact.pos,
                frame=mjx_data.contact.frame,
                includemargin=mjx_data.contact.includemargin,
                solref=mjx_data.contact.solref,
                solimp=mjx_data.contact.solimp,
                solreffriction=mjx_data.contact.solreffriction,
                dim=mjx_data.contact.dim,
                geom1=mjx_data.contact.geom1,
                geom2=mjx_data.contact.geom2,
                geom=mjx_data.contact.geom,
                efc_address=mjx_data.contact.efc_address,
                link_idx=jax.Array(),
                elasticity=jax.Array(),
            ),
        )
        brax_rollout.append(brax_state)
    data = jnp.array(data)
    xdata = jnp.array(xdata)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_states"), data)
    jnp.save(os.path.join(dial_config.output_dir, f"{timestamp}_predictions"), xdata)

    # save the html file and run the server
    import flask

    app = flask.Flask(__name__)
    webpage = html.render(
        env.system, brax_rollout, 1080, True
    )

    with open(
        os.path.join(dial_config.output_dir, f"{timestamp}_brax_visualization.html"),
        "w",
    ) as f:
        f.write(webpage)

    @app.route("/")
    def index():
        return webpage

    app.run(port=5000)


if __name__ == "__main__":
    main()
