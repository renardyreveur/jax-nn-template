import jax.numpy as jnp
from jax import vmap

from modules.model.activations import softmax


# One hot encoding
def one_hot(target, len_pos):
    return jnp.zeros(len_pos).at[target].set(1)


# Cross Entropy single example
def cross_entropy(output, target):
    output = softmax(output)
    target = one_hot(target, 10)
    return - jnp.sum(jnp.dot(target, jnp.log(output)))


# Batched Cross Entropy loss with 'mean' reduction
def batch_cross_entropy(params, model, x, y):
    preds, params = model(x, params=params)
    loss_fn = vmap(cross_entropy)
    return jnp.mean(loss_fn(preds, y))
