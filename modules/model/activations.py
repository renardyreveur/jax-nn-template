import jax
import jax.numpy as jnp


# import jax.scipy as jscipy

def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(x >= 0, 1 / (1 + jnp.exp(-x)), jnp.exp(x) / (1 + jnp.exp(x)))


def hard_swish(in_x: jnp.ndarray) -> jnp.ndarray:
    return in_x * jnp.maximum(0, in_x + 3) / 6


def tanh(in_x: jnp.ndarray) -> jnp.ndarray:
    return jnp.tanh(in_x)


def relu(in_x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0, in_x)


def gelu(in_x: jnp.ndarray, approx: bool = False) -> jnp.ndarray:
    if approx:
        return 0.5 * in_x * (1.0 + tanh(jnp.sqrt(2 / jnp.pi) * (in_x + 0.044715 * jnp.power(in_x, 3))))
    else:
        return 0.5 * in_x * (1.0 + jax.lax.erf(in_x / jnp.sqrt(2)))
        # return in_x * jscipy.stats.norm.cdf(in_x)  # Creates nan when onnx converted


# Multidimensional numerically stable softmax in JAX
def softmax(in_x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    # Subtract max for numerical stability
    x = in_x - jnp.amax(in_x, axis=axis, keepdims=True)
    # Exponentiate x
    x = jnp.exp(x)
    # Sum exponential
    x_sum = jnp.sum(x, axis=axis, keepdims=True)
    # Divide x by sum
    p = x / x_sum
    return p
