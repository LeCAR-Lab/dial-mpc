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


def test_get_foot_step():
    import matplotlib.pyplot as plt

    # Parameters
    gait_phase = jnp.array([0.0, 0.5, 0.5, 0.0])
    duty_ratio, cadence, amplitude = jnp.array([0.45, 2, 0.08])

    time = jnp.linspace(0, 2, 1000)

    zs = []

    for t in time:
        zs.append(get_foot_step(duty_ratio, cadence, amplitude, gait_phase, t))

    zs1 = jnp.array(zs)

    plt.plot(time, zs1)
    plt.show()

    zs2 = get_foot_step(duty_ratio, cadence, amplitude, gait_phase, time)

    plt.plot(time, zs2.T)
    plt.show()

    assert jnp.allclose(zs1, zs2.T)

if __name__ == "__main__":
    test_get_foot_step()
