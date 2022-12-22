import json
import logging.config
import sys
from datetime import datetime
from pathlib import Path
from shutil import copytree, ignore_patterns

import yaml


# Initialization function
def parse_config(args):
    # Read YAML file as dictionary
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError("Configuration YAML file not found!")
    with open(config_path, mode='rt') as cfile:
        try:
            # config = yaml.safe_load(cfile)
            config = yaml.unsafe_load(cfile)
        except yaml.YAMLError:
            raise yaml.YAMLError("Configuration YAML file is malformed, please check!")

    # Create Save Directory
    config = create_save_directory(config)

    # Copy model folder to save_dir
    copytree(Path("./modules/model"), Path(config['save_dir'], "model"), ignore=ignore_patterns("__init__.py"))

    # Setup Logging
    with open("logger/logger_config.json", 'r') as f:
        log_config = json.load(f)
    log_config["handlers"]["file_handler"]["filename"] = Path(config['log_dir'], log_config['handlers']['file_handler']['filename'])
    logging.config.dictConfig(log_config)

    # To suppress JAX internal absl debug logs, turn on if needed!
    if 'absl.logging' in sys.modules:
        import absl.logging
        absl.logging.set_verbosity('info')
        absl.logging.set_stderrthreshold('info')
    for mod in ['jax', 'boto3', 'botocore', 'urllib3', 'git', 'huggingface/tokenizers']:
        logging.getLogger(mod).setLevel(logging.CRITICAL)

    return config


# Save Directory
def create_save_directory(config):
    # Get run_id from experiment tracking details, if not use current datetime
    experiment_name = config['name']
    run_id = config["track_experiment"]["config"]["run_id"] if config["track_experiment"]["track"]\
        else datetime.now().strftime(r'%m%d_%H%M%S')

    # Set save directory where the trained models and logs will be
    save_dir = Path(config['trainer']['save_dir'])
    model_dir = save_dir / 'models' / experiment_name / run_id
    log_dir = save_dir / 'log' / experiment_name / run_id

    # Create directory for saving weight checkpoints and log.
    exist_ok = True  # run_id == ''
    model_dir.mkdir(parents=True, exist_ok=exist_ok)
    log_dir.mkdir(parents=True, exist_ok=exist_ok)

    # Save config file to the checkpoint dir
    with open(Path(model_dir / 'config.yaml'), 'w') as file:
        yaml.dump(config, file, indent=4, default_flow_style=False)
    config.update({"save_dir": str(model_dir), "log_dir": str(log_dir)})
    return config
