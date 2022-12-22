import jax.numpy as jnp


def accuracy(model, params, batch, y):
    pred, _ = model(batch, params=params)
    pred = jnp.argmax(pred, axis=-1)
    assert pred.shape[0] == y.shape[0]
    correct = 0
    correct += jnp.sum(pred == y)
    return correct / pred.shape[0]
