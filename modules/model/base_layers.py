import jax.numpy as jnp
from jax import lax
from jax import random

from modules.model.activations import softmax
from utils import get_params


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


def self_attention(in_x: jnp.ndarray,
                   dim: int, num_heads: int, mask: jnp.ndarray = None,
                   seed: int = 0, params: [dict] = None) -> (jnp.ndarray, [dict]):
    """ Self-Attention Layer
    :param in_x: Input array
    :param dim: Query, Key, Value Dimensions
    :param num_heads: Number of heads in multi-head attention
    :param mask: Mask to apply to attention scores
    :param seed: Weight initialization seed
    :param params: Layer Parameters
    :return: Output array, Parameters
    """
    # Generate Query, Key, Value from input
    qkv, p0 = linear(in_x, dim * 3, seed=seed, params=get_params(params, 0))

    # Split into heads
    b, s, _ = in_x.shape
    queries = qkv[:, :, :dim].reshape(b, s, num_heads, dim // num_heads)
    keys = qkv[:, :, dim:dim * 2].reshape(b, s, num_heads, dim // num_heads)
    values = qkv[:, :, dim * 2:].reshape(b, s, num_heads, dim // num_heads)

    # Calculate attention scores (using query and key) - scaled dot product
    raw_attn_scores = jnp.einsum("i j l m, i o l m->i l j o", queries, keys, optimize='greedy') / jnp.sqrt(
        dim // num_heads)
    if mask is not None:
        raw_attn_scores = raw_attn_scores + mask * -1e9
    attn_scores = softmax(raw_attn_scores, axis=-1)

    # Get sum of weighted values as output
    out = jnp.einsum("i j l m, i m j k -> i l j k", attn_scores, values, optimize='greedy')

    # Concat Heads
    out = out.reshape(*out.shape[:2], -1)
    output, p1 = linear(out, dim, seed=seed, params=get_params(params, 1))

    return output, [p0, p1]
