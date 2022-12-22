import logging
import pickle
import random
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from logger.logger import log_newline
from utils import progress

logger = logging.getLogger("Trainer")
logger.newline = types.MethodType(log_newline, logger)


class Trainer:
    def __init__(self, experiment, tracker, config):
        self.training_dataloader = experiment.dataloader.train
        self.validation_dataloader = experiment.dataloader.val
        self.loss_fn = experiment.loss_fn
        self.model = experiment.model.model
        self.model_params = experiment.model.params
        self.optimizer = experiment.optimizer.optimizer
        self.optimizer_params = experiment.optimizer.params
        self.metrics = experiment.metrics

        self.tracker = tracker
        self.config = config

    def train(self):
        @jax.jit
        def step(params, opt_params, batch, labels):
            loss_val, grads = jax.value_and_grad(self.loss_fn)(params, self.model, batch, labels)
            updates, optimizer_params = self.optimizer.update(grads, opt_params, params)
            model_params = optax.apply_updates(params, updates)
            return loss_val, model_params, optimizer_params

        # ----------------- TRAINING LOOP -----------------
        for epoch in range(self.config['trainer']['epochs']):
            logger.newline()
            logger.info(f"---- STARTING EPOCH {epoch + 1} ----")
            start = time.time()

            loss = 0
            metric_input = {"data": [], "labels": []}
            for batch_idx, (data, label) in enumerate(self.training_dataloader):
                # Update step
                loss_iter, self.model_params, self.optimizer_params = step(self.model_params, self.optimizer_params,
                                                                           data, label)
                loss += loss_iter

                # Collect data for metric calculations and log intermittent results
                if batch_idx % int(jnp.sqrt(self.training_dataloader.batch_size)) == 0:
                    met_idx = random.choice(range(data.shape[0]))
                    metric_input["data"].append(jnp.take(data, met_idx, axis=0))
                    metric_input["labels"].append(jnp.take(label, met_idx, axis=0))
                    # Print Loss
                    logger.info(
                        f"Epoch {epoch + 1} {progress(self.training_dataloader, batch_idx)} -- Loss: {loss_iter}")

            # Post-epoch jobs (metrics, logs, validation etc.)
            metrics = {}
            for k, v in self.metrics.items():
                metrics.update({k: float(v(self.model,
                                           self.model_params,
                                           jnp.stack(metric_input['data'], axis=0),
                                           jnp.stack(metric_input['labels'], axis=0)))})
            for k, v in metrics.items():
                logger.info(f"Epoch {epoch + 1} [train] {k} @ {v} ")

            # For WandB
            wlog = metrics.copy()
            loss /= len(self.validation_dataloader)
            wlog.update({"loss": float(loss)})

            # ----------------- VALIDATION -----------------
            # Validate on Test set
            if self.validation_dataloader is not None:
                val_loss = 0
                val_metrics = {k: 0 for k in self.metrics.keys()}
                for batch_idx, (data, label) in enumerate(self.validation_dataloader):
                    val_loss += self.loss_fn(self.model_params, self.model, data, label)

                    # Calculate metrics
                    for k, v in self.metrics.items():
                        val_metrics[k] += float(v(self.model,
                                                  self.model_params,
                                                  jnp.stack(metric_input['data'], axis=0),
                                                  jnp.stack(metric_input['labels'], axis=0)))
                # Add results to log
                val_loss /= len(self.validation_dataloader)
                val_metrics = {k: v / len(self.validation_dataloader) for k, v in val_metrics.items()}
                logger.info(f"Epoch {epoch + 1} [test] avg Loss: {val_loss}")
                for k, v in val_metrics.items():
                    logger.info(f"Epoch {epoch + 1} [test] {k} @ {v} ")
                wlog.update({"val_loss": float(val_loss)})
                wlog.update(**{'val_' + k: v for k, v in val_metrics.items()})

            end = time.time()
            logger.info(f"---- Epoch {epoch + 1} took {end - start:.2f} seconds to complete! ----")
            if self.config['track_experiment']:
                self.tracker.track(wlog)

            # Save checkpoint every save period
            if (epoch + 1) % self.config['trainer']['save_period'] == 0:
                with open(Path(self.config['save_dir'], f"checkpoint-epoch-{epoch + 1}.params"), 'wb') as f:
                    pickle.dump(self.model_params, f)
