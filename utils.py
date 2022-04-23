import collections
import json
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path
from shutil import ignore_patterns, copytree

import jax.numpy as jnp
from jax import random

logger = logging.getLogger("setup")


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


# Randomly initialized sized array creator
def randn(key, size):
    key, subkey = random.split(key)
    return key, random.normal(subkey, shape=size)


# Training progress bar
def progress(data_loader, b_idx):
    base = '[{}/{} ({:.0f}%)]'
    total = len(data_loader)
    current = b_idx
    return base.format(current, total, 100.0 * current / total)


# Get model parameter count
def parameter_count(params):
    params = [parameter_count(x) if isinstance(x, list) else sum([jnp.prod(jnp.asarray(v.shape)) for _, v in x.items()])
              for x in params]
    return sum(params)


# getattr checker on modules
def get_training_objects(module, name, msg):
    try:
        t_object = getattr(module, name)
        return t_object
    except AttributeError:
        logger.error(f"No such {msg} called {name}")
        return -1


# Logger filter
class Fmt_Filter(logging.Filter):
    def filter(self, record):
        record.levelname = '%s]' % record.levelname
        return True


# Blank line logging
def log_newline(self):
    root_logger = logging.getLogger()
    console_h = root_logger.handlers[-1]
    blank_h = logging.getLogger("blank").handlers[0]

    # Switch handler, output a blank line
    root_logger.removeHandler(console_h)
    root_logger.addHandler(blank_h)
    root_logger.info('')

    # Switch back
    root_logger.removeHandler(blank_h)
    root_logger.addHandler(console_h)


def init_train(args):
    # Get training configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError("Configuration JSON file not found!")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError:
        raise json.JSONDecodeError("Configuration JSON file is malformed, please check!")

    # Create save directory
    root_dir = config['training']['save_dir']
    title = config['title']
    identifier = config['wandb']['id'] if config['track_experiment'] else datetime.now().strftime(r'%m%d_%H%M%S')
    save_dir = Path(root_dir, title, identifier)
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Copy train configuration to save_dir
    with open(Path(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    config['training']['save_dir'] = save_dir

    # Copy model folder to save_dir
    copytree(Path("./model"), Path(save_dir, "model"), ignore=ignore_patterns("__init__.py"))

    # Setup logging
    with open("logger_config.json", 'r') as f:
        log_config = json.load(f)
    log_config['handlers']['file_handler']['filename'] = Path(save_dir,
                                                              log_config['handlers']['file_handler']['filename'])
    logging.config.dictConfig(log_config)

    # To suppress JAX internal absl debug logs, turn on if needed!
    if 'absl.logging' in sys.modules:
        import absl.logging
        absl.logging.set_verbosity('info')
        absl.logging.set_stderrthreshold('info')

    return config
