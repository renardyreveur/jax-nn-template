import jax.numpy as jnp


def layer_norm(in_x, dim, params=None):
    """ Layer Normalization

    :param in_x: Input array
    :param dim: Dimension to carry out normalization
    :param params: Layer Parameters
    :return: Normalized output, Layer Parameters
    """
    if params is None:
        params = {"weight": jnp.ones(in_x.shape[dim]), "bias": jnp.zeros(in_x.shape[dim])}
    if in_x.shape[dim] != params['weight'].shape[0]:
        raise ValueError(f"Parameter Shape doesn't match layer! {in_x.shape[dim]} and {params['weight'].shape}")
    eps = 0.00001
    mean = jnp.mean(in_x, axis=dim, keepdims=True)
    var = jnp.var(in_x, axis=dim, keepdims=True)
    x = (in_x - mean) / jnp.sqrt(var + eps)
    w = jnp.reshape(params['weight'], (1,) * dim + (-1,) + (1,) * (len(in_x.shape) - 1 - dim))
    b = jnp.reshape(params['bias'], (1,) * dim + (-1,) + (1,) * (len(in_x.shape) - 1 - dim))
    return w * x + b, params
