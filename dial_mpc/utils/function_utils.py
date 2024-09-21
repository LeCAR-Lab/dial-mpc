from brax import math

import jax.numpy as jnp


def global_to_body_velocity(v, q):
    """Transforms global velocity to body velocity."""
    # rotate v by inverse of q
    return math.inv_rotate(v, q)


def body_to_global_velocity(v, q):
    """Transforms body velocity to global velocity."""
    return math.rotate(v, q)
