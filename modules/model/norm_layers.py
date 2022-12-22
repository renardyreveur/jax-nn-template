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
    new_shape = [1] * len(in_x.shape)
    new_shape[dim] = -1
    w = params['weight'].reshape(new_shape)
    b = params['bias'].reshape(new_shape)
    return jnp.multiply(x, w) + b, params
