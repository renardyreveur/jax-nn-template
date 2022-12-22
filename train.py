import argparse
import logging
import types

from experiment import Experiment, Tracker, Trainer
from logger.logger import log_newline
from parse_config import parse_config


def main(config):
    # Logging
    logger = logging.getLogger("main")
    logger.newline = types.MethodType(log_newline, logger)
    logger.info(f"Start Training! [{config['name']}]")

    # Set up Experiment (Model, Data, Loss_fn, Optimizer, etc.)
    experiment = Experiment(config)

    # Set up Experiment Tracker
    tracker = Tracker(config)

    # Instantiate the Trainer and train
    trainer = Trainer(experiment, tracker, config)
    trainer.train()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAX Template")
    parser.add_argument('-c', '--config', default="config.yaml", type=str,
                        help="Path to Configuration YML file (default: config.yaml)")
    args = parser.parse_args()

    # Initialize training configurations
    cfgs = parse_config(args)

    # Here we go!
    main(cfgs)
