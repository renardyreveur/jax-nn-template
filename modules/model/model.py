import jax.numpy as jnp

from modules.model.activations import gelu
from modules.model.base_layers import conv2d, linear
from modules.model.norm_layers import layer_norm
from utils import get_params


def downsample(in_x, in_dim, out_dim, params=None):
    x, p0 = layer_norm(in_x, 1, get_params(params, 0))
    x, p1 = conv2d(x, in_dim, out_dim, kernel_size=(2, 2), stride=(2, 2), seed=41, params=get_params(params, 1))
    return x, [p0, p1]


def convnext_block_alt(in_x, dim, params=None):
    """
    Doesn't work with the final onnx conversion
    """
    x, p0 = conv2d(in_x, dim, dim, kernel_size=(7, 7), padding=((3, 3), (3, 3)), groups=dim,
                   seed=42, params=get_params(params, 0))
    x = jnp.transpose(x, (0, 2, 3, 1))
    x, p1 = layer_norm(x, 3, get_params(params, 1))
    x, p2 = linear(x, dim * 2, seed=43, params=get_params(params, 2))
    x = gelu(x)
    x, p3 = linear(x, dim, seed=44, params=get_params(params, 3))
    x = jnp.transpose(x, (0, 3, 1, 2))
    # TODO: Add DropPath
    return x, [p0, p1, p2, p3]


def convnext_block(in_x, dim, params=None):
    x, p0 = conv2d(in_x, dim, dim, kernel_size=(7, 7), padding=((3, 3), (3, 3)), groups=dim,
                   seed=42, params=get_params(params, 0))
    x, p1 = layer_norm(x, 1, get_params(params, 1))
    x, p2 = conv2d(x, dim, dim * 2, kernel_size=(1, 1), seed=43, params=get_params(params, 2))
    x = gelu(x)
    x, p3 = conv2d(x, dim * 2, dim, kernel_size=(1, 1), seed=44, params=get_params(params, 3))
    # TODO: Add DropPath
    return x, [p0, p1, p2, p3]


def mnist_model(in_x, feat_dims, params=None):
    new_params = []
    # Stem
    x, p0 = conv2d(in_x, 1, feat_dims[0], kernel_size=(4, 4), stride=(4, 4),
                   seed=12, params=get_params(params, 0))
    x, p1 = layer_norm(x, 1, get_params(params, 1))
    new_params.append(p0)
    new_params.append(p1)

    # Body
    for i in range(len(feat_dims)):
        x, p2 = convnext_block(x, dim=feat_dims[i], params=get_params(params, 2 + i * 2))
        new_params.append(p2)
        if i != len(feat_dims) - 1:
            x, p3 = downsample(x, in_dim=feat_dims[i], out_dim=feat_dims[i + 1], params=get_params(params, 3 + i * 2))
            new_params.append(p3)

    features, p4 = layer_norm(jnp.mean(x, axis=(-2, -1)), dim=1, params=get_params(params, len(feat_dims) * 2 + 1))
    new_params.append(p4)

    # Head
    proj, p5 = linear(features, 64, params=get_params(params, len(feat_dims) * 2 + 2))
    out, p6 = linear(proj, 10, params=get_params(params, len(feat_dims) * 2 + 3))
    new_params.append(p5)
    new_params.append(p6)

    return out, new_params
