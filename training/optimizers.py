from functools import partial

import jax
import jax.numpy as jnp


# SGD
from jax import value_and_grad


def sgd(w, g, op_params=None, lr=0.001, **kwargs):
    def init_sgd_params(params):
        params = [init_sgd_params(x) if isinstance(x, list)
                  else {k: {"step": 0} for k, v in x.items()}
                  for x in params]
        return params

    if op_params is None:
        return init_sgd_params(w)
    return w - lr * g, op_params


# Adam with weight decay
def adamw(w, g, op_params=None, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001):
    def init_adamw_params(params):
        params = [init_adamw_params(x) if isinstance(x, list)
                  else {k: {"m0": jnp.zeros_like(v), "v0": jnp.zeros_like(v), "step": 0} for k, v in x.items()}
                  for x in params]
        return params

    if op_params is None:
        return init_adamw_params(w)
        # return [{k: {"m0": 0, "v0": 0, "step": 0} for k, v in layer.items()} for layer in w]

    # Momentum
    # Gradient direction is smoothed by exponentially weighing the moving averages
    op_params["m0"] = betas[0] * op_params["m0"] + (1 - betas[0]) * g
    # RMSProp
    # Gradient magnitude is smoothed such that it slows down near flats, and doesn't flick off at suboptimal gradients
    op_params["v0"] = betas[1] * op_params["v0"] + (1 - betas[1]) * (g ** 2)

    # Estimation bias correction
    mt_hat = op_params["m0"] / (1 - betas[0] ** op_params['step'])
    vt_hat = op_params["v0"] / (1 - betas[1] ** op_params['step'])

    # Weight decay is not the same as L2 regularization when not standard SGD ->
    # substituting g with (g + wd*w) which comes from adding sum of squared weights to loss
    # is not equal to weight decay when gradients are altered (such as momentum, etc.)
    new_w = w - (lr * mt_hat / (jnp.sqrt(vt_hat) + eps) + weight_decay * w)
    return new_w, op_params


# ----- Update function -----
def update_optim_params(params):
    [update_optim_params(x) if isinstance(x, list) else [v.update({"step": v['step'] + 1}) for k, v in x.items()]
     for x in params]


@partial(jax.jit, static_argnums=(2,))
# What if loss_fn, optimizer partial functions break tracing? Separate functions for each?
def update(in_data, loss_fn, model, params, optimizer, optimizer_params, **kwargs):
    # Calculate loss and gradients of loss w.r.t params
    loss, gradient = value_and_grad(loss_fn, argnums=0)(params, model, *in_data)

    # Step counter in optimizer_params update
    update_optim_params(optimizer_params)

    # Updatable parameters
    num_up_params = len(optimizer_params)

    # Update parameters with specified optimization algorithm
    updated_params = jax.tree_map(optimizer, params[:num_up_params], gradient[:num_up_params], optimizer_params)

    # Reformat PyTree with tree_transpose
    new_params, new_oparams = jax.tree_util.tree_transpose(outer_treedef=jax.tree_structure(params[:num_up_params]),
                                                           inner_treedef=jax.tree_structure(
                                                               (0, {'m0': 0, 'v0': 0, 'step': 0})),
                                                           pytree_to_transpose=updated_params)
    new_oparams = jax.tree_util.tree_transpose(outer_treedef=jax.tree_structure({'m0': 0, 'v0': 0, 'step': 0}),
                                               inner_treedef=jax.tree_structure(params[:num_up_params]),
                                               pytree_to_transpose=new_oparams)

    # If input was single parameter wrapped in a tuple, reduce that tuple when returning
    if len(new_params) == 1 and isinstance(new_params[0], list):
        new_params, new_oparams = new_params[0], new_oparams[0]

    return loss, new_params, new_oparams
