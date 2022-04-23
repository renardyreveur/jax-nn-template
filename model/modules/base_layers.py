import jax.numpy as jnp
from jax import lax
from jax import random

from .activations import softmax


def linear(in_x, proj_dim, bias=True, seed=0, params=None):
    """ Fully connected linear layer

    :param in_x: Input array
    :param int proj_dim: Projection dimension
    :param bool bias: Whether to carry bias array or not
    :param int seed: Parameter initialization seed
    :param params: Layer parameters
    :return: Projected array, Layer Parameters
    """
    if params is None:
        k = random.PRNGKey(seed)
        params = {}
        for i in ['weight', 'bias'] if bias else ['weight']:
            k, subkey = random.split(k)
            shape = (in_x.shape[-1], proj_dim) if i == "weight" else (proj_dim,)
            p = random.normal(subkey, shape=shape)
            p *= jnp.sqrt(2 / proj_dim)
            params.update({i: p})

    if params['weight'].shape[-1] != proj_dim or params['weight'].shape[-2] != in_x.shape[-1]:
        raise ValueError(
            f"Parameter Shape doesn't match layer! {params['weight']} is not {in_x.shape[-1]} x {proj_dim}")

    return in_x @ params['weight'] + (params['bias'] if bias else 0), params


def conv2d(in_x, in_chns, out_chns, kernel_size, padding="VALID", groups=1, stride=(1,) * 2, bias=False,
           seed=0, params=None):
    """ 2D Convolution Layer

    :param in_x: Input array
    :param int in_chns: Input channel length
    :param int out_chns: Output channel length
    :param tuple(int, int) kernel_size: Convolution kernel size
    :param padding: Padding to apply at each spatial dimension [(low, high), (low, high)]
    :param int groups: Number of channel groups
    :param stride: Kernel window stride
    :param bool bias: Carry bias array or not
    :param int seed: Parameter weight initialization seed
    :param params: Layer Parameters
    :return: Convolved output array, Layer Parameters
    """
    if params is None:
        k = random.PRNGKey(seed)
        params = {}
        for i in (['weight', 'bias'] if bias else ['weight']):
            k, subkey = random.split(k)
            shape = (out_chns, in_chns // groups, *kernel_size) if i == "weight" else (out_chns,)
            p = random.uniform(subkey, minval=-jnp.sqrt(groups / (in_chns * jnp.prod(jnp.array(kernel_size)))),
                               maxval=jnp.sqrt(groups / (in_chns * jnp.prod(jnp.array(kernel_size)))),
                               shape=shape)
            params.update({i: p})

    conv = lax.conv_general_dilated(in_x, params['weight'], stride, padding, feature_group_count=groups)

    if bias:
        conv += jnp.reshape(params['bias'], (1, out_chns, 1, 1))

    return conv, params


def attention(in_x, dim, keys=None, values=None, seed=0, params=None):
    """ Attention / Self-Attention Layer

    :param in_x: Input array
    :param dim: Query, Key, Value Dimensions
    :param keys: If provided, use external key (encoder-decoder attention), else self-attention
    :param values: If provided, used external value
    :param seed: Weight initialization seed
    :param params: Layer Parameters
    :return: Output array, Parameters
    """
    if params is None:
        k = random.PRNGKey(seed)
        params = {}
        for i in ['w_query', 'w_key', 'w_val']:
            k, subkey = random.split(k)
            shape = (in_x.shape[-1], dim)
            p = random.normal(subkey, shape=shape)
            p *= jnp.sqrt(2 / dim)
            params.update({i: p})

    # Generate Query, Key, Value from input
    queries = in_x @ params['w_query']
    if keys is None:
        keys = in_x @ params['w_key']
        values = in_x @ params['w_val']

    # Calculate attention scores (using query and key) - scaled dot product
    attn_scores = softmax(1 / jnp.sqrt(queries.shape[-1]) * (queries @ jnp.transpose(keys, axes=[0, 2, 1])))

    # Get sum of weighted values as output
    return attn_scores @ values, params
