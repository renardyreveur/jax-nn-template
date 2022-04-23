import argparse
import itertools
import logging
import pickle
import random
import time
import types
from pathlib import Path

import jax.numpy as jnp
import jax.tree_util

import data_loader as dl_module
import model as model_module
import training.loss as loss_module
import training.metrics as metrics_module
import training.optimizers as opt_module
import wandb
from utils import init_train, get_training_objects, parameter_count, progress, log_newline, flatten


def main(config):
    if config['track_experiment']:
        wandb.init(config=flatten(config), **config['wandb'])
    logger = logging.getLogger("main")
    logger.newline = types.MethodType(log_newline, logger)
    logger.info(f"Start Training! [{config['title']}]")

    # Get DataLoader, Model, Optimizer, Loss function
    dataloader = get_training_objects(dl_module, config['data_loader']['loader'], "dataloader")
    testloader = None
    if 'test_loader' in config['data_loader'].keys():
        testloader = get_training_objects(dl_module, config['data_loader']['test_loader'], "dataloader")
    model = get_training_objects(model_module, config['model_struct']['model'], "model")
    optimizer = get_training_objects(opt_module, config['training']['optimizer'], "optimizer")
    loss_fn = get_training_objects(loss_module, config['training']['loss_fn'], "loss_fn")
    if -1 in [dataloader, testloader, model, optimizer, loss_fn]:
        return -1

    # Prime them with the configuration provided in the JSON file
    try:
        dataloader = dataloader(**config['data_loader']['args'])
        if 'test_loader' in config['data_loader'].keys():
            testloader = testloader(**config['data_loader']['test_args'])
        config['model_struct']['args'] = {k: tuple(v) if isinstance(v, list) else v for k, v in
                                          config['model_struct']['args'].items()}
        config['training']['optimizer_args'] = {k: tuple(v) if isinstance(v, list) else v for k, v in
                                                config['training']['optimizer_args'].items()}
        model = jax.tree_util.Partial(model, **config['model_struct']['args'])
        optimizer = jax.tree_util.Partial(optimizer, **config['training']['optimizer_args'])
        loss_fn = jax.tree_util.Partial(loss_fn)
    except Exception:
        logger.exception("Check configuration parameters!")
    logger.info("Dataloader, Model, Optimizer, Loss function loaded!")

    # Initialize model parameters
    dummy_sample, _ = dataloader.dataset[0]
    _, model_params = model(jnp.expand_dims(dummy_sample, 0))
    logger.info(f"The model has {parameter_count(model_params)} parameters")
    logger.newline()

    # Check if checkpoint weights are provided
    if "checkpoint" in config['training'].keys() and config['training']['checkpoint'] != "":
        ckpt_path = Path(config['training']['checkpoint'])
        if not ckpt_path.exists():
            logger.debug("Checkpoint path provided in configuration doesn't exist!")
            logger.debug("!!!Continuing with new weights!!!")
            logger.newline()
        else:
            with open(ckpt_path, 'rb') as f:
                pretrained_params = pickle.load(f)
            if jax.tree_util.tree_structure(model_params) != jax.tree_util.tree_structure(pretrained_params) or \
                    jax.tree_map(lambda p: p.shape, model_params) != jax.tree_map(lambda p: p.shape, pretrained_params):
                logger.debug("Provided checkpoint file does not match model dimensions!")
                logger.debug("!!!Continuing with new weights!!!")
                logger.newline()
            else:
                model_params = pretrained_params

    # Initialize optimizer parameters
    opt_params = optimizer(model_params, None)

    # Training loop
    for epoch in range(config['training']['epochs']):
        logger.newline()
        logger.info(f"---- STARTING EPOCH {epoch + 1} ----")
        start = time.time()
        loss = 0
        for batch_idx, (data, label) in enumerate(dataloader):
            # Gradient Descent and Parameter updates
            loss, model_params, opt_params = opt_module.update(
                in_data=(data, label),
                loss_fn=loss_fn,
                model=model,
                params=model_params,
                optimizer=optimizer,
                optimizer_params=opt_params,
            )

            if batch_idx % int(jnp.sqrt(dataloader.batch_size)) == 0:
                logger.info(f"Epoch {epoch + 1} {progress(dataloader, batch_idx)} -- Loss: {loss}")

        # Post-epoch jobs (metrics, logs, validation etc.)
        metrics = [0] * len(config['training']['metrics'])
        for i, m in enumerate(config['training']['metrics']):
            met = getattr(metrics_module, m)
            for _ in range(3):
                n = random.randint(0, len(dataloader)-1)
                x, y = next(itertools.islice(dataloader, n, n + 1))
                metrics[i] += met(model, model_params, x, y)
        metrics = [x / 3 for x in metrics]
        metrics = {k: v for k, v in zip(config['training']['metrics'], metrics)}
        for k, v in metrics.items():
            logger.info(f"Epoch {epoch + 1} [train] {k} @ {v} ")

        # For WandB
        wlog = metrics.copy()
        wlog.update({"loss": loss})
        wlog.update({"epoch": epoch+1})

        # Validate on Test set
        if testloader is not None:
            val_loss = 0
            val_metrics = [0] * len(config['training']['metrics'])
            for batch_idx, (data, label) in enumerate(testloader):
                val_loss += loss_fn(model_params, model, data, label)
                for i, m in enumerate(config['training']['metrics']):
                    met = getattr(metrics_module, m)
                    val_metrics[i] += met(model, model_params, data, label)
            val_loss /= len(testloader)
            val_metrics = [x / len(testloader) for x in val_metrics]
            val_metrics = {k: v for k, v in zip(config['training']['metrics'], val_metrics)}
            logger.info(f"Epoch {epoch + 1} [test] avg Loss: {val_loss}")
            for k, v in val_metrics.items():
                logger.info(f"Epoch {epoch + 1} [test] {k} @ {v} ")
            wlog.update({"val_loss": val_loss})
            wlog.update(**{'val_' + k: v for k, v in val_metrics.items()})

        end = time.time()
        logger.info(f"---- Epoch {epoch + 1} took {end - start:.2f} seconds to complete! ----")
        if config['track_experiment']:
            wandb.log(wlog)

        # Save checkpoint every save period
        if (epoch+1) % config['training']['save_period'] == 0:
            with open(Path(config['training']['save_dir'], f"checkpoint-epoch_{epoch+1}.params"), 'wb') as f:
                pickle.dump(model_params, f)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAX NN Template")
    parser.add_argument('-c', '--config', default=".config.json", type=str,
                        help="Path to Configuration JSON file (default: config.json)")
    args = parser.parse_args()

    # Initialize training configurations
    cfgs = init_train(args)

    # Start training!
    main(cfgs)
