import collections
import logging

import jax
from jax import random

logger = logging.getLogger("setup")


# Randomly initialized sized array creator
def randn(key, size):
    key, subkey = random.split(key)
    return key, random.normal(subkey, shape=size)


# Get parameter by index
def get_params(params, idx):
    return None if params is None else params[idx]


# Get model parameter count
def parameter_count(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


# Flatten dictionary
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# getattr checker on modules
def get_training_objects(module, name, msg):
    try:
        t_object = getattr(module, name)
        return t_object
    except AttributeError:
        logger.error(f"No such {msg} called {name}")
        raise AttributeError


# Training progress bar
def progress(data_loader, b_idx):
    base = '[{}/{} ({:.0f}%)]'
    total = len(data_loader)
    current = b_idx
    return base.format(current, total, 100.0 * current / total)
