import jax
import jax.numpy as jnp
import jax.scipy as jscipy


def sigmoid(x):
    return jnp.where(x >= 0, 1 / (1 + jnp.exp(-x)), jnp.exp(x) / (1 + jnp.exp(x)))


def hard_swish(in_x):
    return in_x * jnp.maximum(0, in_x + 3) / 6


def tanh(in_x):
    return jnp.tanh(in_x)


def gelu(in_x, approx=True):
    # if approx:
    return 0.5 * in_x * (1.0 + tanh(jnp.sqrt(2 / jnp.pi) * (in_x + 0.044715 * jnp.power(in_x, 3))))
    # else:
    #     return 0.5 * in_x * (1.0 + jax.lax.erf(in_x/jnp.sqrt(2)))
    # return in_x * jscipy.stats.norm.cdf(in_x)


def softmax(in_x):
    return jnp.divide(jnp.exp(in_x), jnp.sum(jnp.exp(in_x)))
