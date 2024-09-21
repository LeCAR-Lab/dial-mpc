from brax import math

import jax.numpy as jnp
import jax


def global_to_body_velocity(v, q):
    """Transforms global velocity to body velocity."""
    # rotate v by inverse of q
    return math.inv_rotate(v, q)


def body_to_global_velocity(v, q):
    """Transforms body velocity to global velocity."""
    return math.rotate(v, q)


@jax.jit
def get_foot_step(duty_ratio, cadence, amplitude, phases, time):
    """
    Compute the foot step height.
    Args:
        amplitude: The height of the step.
        cadence: The cadence of the step (per second).
        duty_ratio: The duty ratio of the step (% on the ground).
        phases: The phase of the step. Warps around 1. (N-dim where N is the number of legs)
        time: The time of the step.
    """

    def step_height(t, footphase, duty_ratio):
        angle = (t + jnp.pi - footphase) % (2 * jnp.pi) - jnp.pi
        angle = jnp.where(duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle)
        clipped_angle = jnp.clip(angle, -jnp.pi / 2, jnp.pi / 2)
        value = jnp.where(duty_ratio < 1, jnp.cos(clipped_angle), 0)
        final_value = jnp.where(jnp.abs(value) >= 1e-6, jnp.abs(value), 0.0)
        return final_value

    h_steps = amplitude * jax.vmap(step_height, in_axes=(None, 0, None))(
        time * 2 * jnp.pi * cadence + jnp.pi,
        2 * jnp.pi * phases,
        duty_ratio,
    )
    return h_steps
