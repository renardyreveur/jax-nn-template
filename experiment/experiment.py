import logging
import pickle
import types
from collections import namedtuple
from pathlib import Path
from typing import Union, Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

import dataloader.dataloader as dl_module
import modules.loss as loss_module
import modules.metrics as metric_module
import modules.model as model_module
from logger.logger import log_newline
from utils import get_training_objects
from utils import parameter_count

logger = logging.getLogger("Experiment")
logger.newline = types.MethodType(log_newline, logger)


class Experiment:
    def __init__(self, config):
        self.config = config
        self.dataloader = self._get_dataloader()
        self.model = self._get_model()
        self.loss_fn = self._get_lossfn()
        self.optimizer = self._get_optimizer()
        self.metrics = self._get_metrics()

    def _get_dataloader(self) -> (dl_module.BaseDataLoader, Union[dl_module.BaseDataLoader, None]):
        """ Get dataloader for training and validation """
        # Get the dataloader class and instantiate it with the config file
        Dataloader = namedtuple('Dataloader', ['train', 'val'])
        dataloader, valloader = None, None
        try:
            dataloader = get_training_objects(dl_module, self.config['dataloader']['type'], "dataloader")
            dataloader = dataloader(**self.config['dataloader']['args'])
        except Exception as e:
            logger.error("Could not instantiate the dataloader!")
            raise e

        # If a test/validation dataset is specified, load it
        if 'validation_loader' in self.config['dataloader'].keys() \
                and self.config['dataloader']['validation_loader'] != "":
            try:
                valloader = get_training_objects(dl_module, self.config['dataloader']['validation_loader'],
                                                 "dataloader")
                valloader = valloader(**self.config['dataloader']['validation_args'])
            except Exception as e:
                logger.error("Could not instantiate the validation dataloader!")
                raise e

        return Dataloader(dataloader, valloader)

    def _get_model(self) -> (Callable, [dict]):
        """ Get the model and its initial parameters """
        Model = namedtuple('Model', ['model', 'params'])
        try:
            model = get_training_objects(model_module, self.config['model']['type'], "model")
            model = jtu.Partial(model, **self.config['model']['args'])
        except Exception as e:
            logger.error("Could not instantiate the model!")
            raise e

        # Initialize model parameters
        dummy_sample, _ = self.dataloader.train.dataset[0]
        _, model_params = model(jnp.expand_dims(dummy_sample, 0))
        logger.info(f"The model has {parameter_count(model_params)} parameters")
        logger.newline()

        # Check if checkpoint weights are provided
        if "trained_weights" in self.config.keys() and self.config['trained_weights'] is not None:
            ckpt_path = Path(self.config['trained_weights'])
            if not ckpt_path.exists():
                logger.debug("Checkpoint path provided in configuration doesn't exist!")
                logger.debug("!!!Continuing with new weights!!!")
                logger.newline()
            else:
                with open(ckpt_path, 'rb') as f:
                    pretrained_params = pickle.load(f)
                if jtu.tree_structure(model_params) != jtu.tree_structure(pretrained_params) or \
                        jax.tree_map(lambda p: p.shape, model_params) != jax.tree_map(lambda p: p.shape,
                                                                                      pretrained_params):
                    logger.debug("Provided checkpoint file does not match model dimensions!")
                    logger.debug("!!!Continuing with new weights!!!")
                    logger.newline()
                else:
                    model_params = pretrained_params
                    logger.info(f"Using pretrained weights from checkpoint file: {ckpt_path}")
        return Model(model, model_params)

    def _get_lossfn(self) -> Callable:
        """ Get the loss function """
        try:
            loss_fn = get_training_objects(loss_module, self.config['loss'], "loss")
        except Exception as e:
            logger.error("Could not instantiate the loss function!")
            raise e
        return loss_fn

    def _get_optimizer(self) -> (optax.GradientTransformation, optax.Params):
        """ Get the optimizer """
        Optimizer = namedtuple('Optimizer', ['optimizer', 'params'])
        try:
            optimizer = get_training_objects(optax, self.config['optimizer']['type'], "optimizer")
            optimizer = optimizer(**self.config['optimizer']['args'])
        except Exception as e:
            logger.error("Could not instantiate the optimizer!")
            raise e

        # Initialize optimizer parameters
        opt_params = optimizer.init(self.model.params)

        return Optimizer(optimizer, opt_params)

    def _get_scheduler(self):
        pass

    def _get_metrics(self):
        """ Get metric functions """
        try:
            metrics = {m: get_training_objects(metric_module, m, "metric") for m in self.config['metrics']}
        except Exception as e:
            logger.error("Could not instantiate the metric functions!")
            raise e
        return metrics
