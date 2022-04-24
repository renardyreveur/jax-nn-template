# JAX Neural Network Development Template
[<img src="https://raw.githubusercontent.com/google/jax/master/images/jax_logo_250px.png" alt="JAX Logo" align="right" height="100">](https://github.com/google/jax)

Neural Network Development and Training template with JAX.


[**Introduction**](#introduction)
| [**Structure**](#structure)
| [**Usage**](#usage)
| [**Test Envs**](#test-envs)
| [**Acknowledgements**](#acknowledgements)

## Introduction

JAX is awesome :heart:; this project aims to provide a starting template for training neural networks built with JAX.

Some features that this template have: 

- **Clear structure**:books: to differentiate between the multiple parts involved in training a neural network (Data-loading, Modelling, Training, etc.)
- **Custom(izable) neural network library**:thinking: with custom(izable) optimizers and update functions
- **`JSON` configuration file**:pencil2: to easily define/switch between training hyper-parameters
- **Checkpoint saving**:white_check_mark: and Training logging
- **[`WandB`](https://wandb.ai/site)** Integration :chart_with_downwards_trend:
- Experimental **`tflite` and `ONNX` conversion** support :arrows_clockwise:

>The template is preloaded with a `MNIST training sample`, so it can be run straight away to see how the cookie :cookie: crumbles (in a good way!)

The project sample doesn't use [`Flax`](https://github.com/google/flax) or [`Haiku`](https://github.com/deepmind/dm-haiku) for modelling the neural network (or any of the optimizers or the update function, etc.), but I think(not tested) it would be possible to combine the libraries with the custom definitions in the template. This is intentional as I wanted to practice designing and understanding the layers involved in building a neural network, and to also have maximum customization powers. The way JAX handles `pytrees` is very helpful in creating user-defined trees to represent various parts of the training process(parameters, etc.).





## Structure

| Directory/Root File | Description                                                                                                                                                                               |
|:-------------------:|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      **data**       | A place where your data resides                                                                                                                                                           |
|   **data_loader**   | `base.py` contains a base class for data loaders<br> Implement a `Dataset` and `Dataloader` in `dataloader.py`                                                                            |
|    **inference**    | `convert_tflite.py` is a script that converts your JAX model and weights <br> into a `tflite` file and if possible, an `onnx` file<br> (Check out the **Usage** section below)            |
|      **model**      | **`modules`** contains basic building blocks used in creating neural networks, very raw at this stage, extend as necessary<br> Use`model.py` to define neural networks to use in training |
|      **saved**      | Empty folder that will be populated with trained weight checkpoints                                                                                                                       |
|    **training**     | This is where you define your **`loss function`**, **`training/validation metrics`**, and **`optimizers`**<br> The *update function* is in `optimizers.py`                                |
|     config.json     | `JSON` file containing hyper-parameters for the training runs                                                                                                                             |
| logger_config.json  | logger configuration file                                                                                                                                                                 |
|       test.py       | Empty file, to be populated with test scripts, etc.                                                                                                                                       |
|      train.py       | THE MAIN FILE - Controls the entire logic of training                                                                                                                                     |
|      utils.py       | Utility file containing various small functions that have been refactored out                                                                                                             |
|  requirements.txt   | Python package requirements file                                                                                                                                                          |

## Usage

The usage of this template is typically separated into three parts: [**Development**](#development), [**Training**](#training), [**Inference**](#inference)

### Development

**Data Loading**
 - In order to use this template, you need to first add training/testing data to the **`data`** folder
 - The data will be fed to the trainer using the data-loader defined in `data_loader/dataloader.py`.
 - Create a dataset class that carries at least the `__getitem__` and `__len__` functions
 - The dataset object is fed into a dataloader that inherits the [`BaseDataLoader`](https://github.com/renardyreveur/jax-nn-template/blob/master/data_loader/base.py)
 - Define a *custom batch collate function* if needed and add it as a parameter to the dataloader (if you want to control this from config.json, create a `getattr` parser as well!)

**Model Creation**
- Try and adopt the functional programming paradigm, especially for JIT.
- Create neural networks to train in `model/model.py` using the base layers in the `model/module` module / custom layers / Flax, Haiku, etc. as **`Python functions`**
- The 'made-from-scratch' models in this template carries the model parameters in a convoluted tree of list and dictionaries. This is fed into the model as a function parameter
- Base layers require a parameter initialization part where it creates initial parameters when none are given
- To make it work with the experimental onnx converter, you might need to use approximations or alternative forms for the same layer (as shown in the template)

**Define Loss, Metric, Optimizer**
- Again as python functions, define your loss functions, metric functions and optimizers inside the `training` folder.
- `vmap` comes in handy here for easily making batch-ful computations
- optimizers can carry parameters as well, the same way as model parameters are handled
- The `update` function in the `training/optimizers.py` can probably handle multiple optimizers and multiple models at once, but fix as necessary

**Tune the main training script**
- Tune the `train.py` main script as necessary for your experimentation



### Training

Once the development phase is over, you are ready to train.
The following are controlled by the configuration JSON file `config.json`, edit to your need

**Hyper-parameters in JSON**:
- WandB logging configurations
- Which model to use (by name), and its arguments
- Which data/test-loader to use (by name), and its arguments
- Which optimizer to use (by name), and its arguments
- Which loss function to use (by name)
- Which metrics to track (by name)
- How many epochs to train for
- Whether to load pre-trained weights before training
- Where to save checkpoints and the interval of doing so

Add more options and change the codebase accordingly for extended use cases.

Once you are set to go, run the following command to start training with your configuration

```shell
python train.py -c config.json
```

Hopefully, it  will produce a console log similar to the following:

```shell
[main INFO]    : 2022-04-25 02:36:14,020 - Start Training! [MNIST_Sample]
[main INFO]    : 2022-04-25 02:36:14,053 - Dataloader, Model, Optimizer, Loss function loaded!
[absl INFO]    : 2022-04-25 02:36:14,058 - Remote TPU is not linked into jax; skipping remote TPU.
[absl INFO]    : 2022-04-25 02:36:14,058 - Unable to initialize backend 'tpu_driver': Could not initialize backend 'tpu_driver'
[absl INFO]    : 2022-04-25 02:36:14,138 - Unable to initialize backend 'tpu': INVALID_ARGUMENT: TpuPlatform is not available.
[main INFO]    : 2022-04-25 02:36:18,151 - The model has 13810 parameters


[main INFO]    : 2022-04-25 02:36:19,369 - ---- STARTING EPOCH 1 ----
[main INFO]    : 2022-04-25 02:36:24,354 - Epoch 1 [0/118 (0%)] -- Loss: 5.887911796569824
[main INFO]    : 2022-04-25 02:36:30,918 - Epoch 1 [22/118 (19%)] -- Loss: 0.8348695635795593
[main INFO]    : 2022-04-25 02:36:37,899 - Epoch 1 [44/118 (37%)] -- Loss: 0.5723302960395813
[main INFO]    : 2022-04-25 02:36:44,871 - Epoch 1 [66/118 (56%)] -- Loss: 0.3618883192539215
[main INFO]    : 2022-04-25 02:36:51,849 - Epoch 1 [88/118 (75%)] -- Loss: 0.36900243163108826
[main INFO]    : 2022-04-25 02:36:58,628 - Epoch 1 [110/118 (93%)] -- Loss: 0.33672940731048584
[main INFO]    : 2022-04-25 02:37:43,137 - Epoch 1 [train] accuracy @ 0.9055989980697632 
[main INFO]    : 2022-04-25 02:37:54,400 - Epoch 1 [test] avg Loss: 0.2996721863746643
[main INFO]    : 2022-04-25 02:37:54,401 - Epoch 1 [test] accuracy @ 0.9099782109260559 
[main INFO]    : 2022-04-25 02:37:54,401 - ---- Epoch 1 took 95.03 seconds to complete! ----

[main INFO]    : 2022-04-25 02:37:54,401 - ---- STARTING EPOCH 2 ----

...
```

As the training progresses, weight checkpoints will be saved under the designated 'save folder' specified in the JSON configuration file.
```text
saved/
│
├── 'title' given in config.json
 ...   ├── datetime / wandb_id depending on configuration
       ...      ├── model/       -   copy of model/ directory when training script was invoked
                ├── checkpoint-epoch_n.params   # n being multiplies of save_period
                ├── ...
                ├── train.log    -   copy of console log as file
                └── config.json  -   copy of config.json when training script was invoked
```


### Inference

This last part deals with exporting the model and weights into something more standardized such as [**ONNX**](https://github.com/onnx/onnx).

The script `inference/convert_tflite.py` converts a given model definition and weights into a tensorflow-lite model file. If it is possible, it also tries to convert the tflite file into an onnx file using [tf2onnx](https://github.com/onnx/tensorflow-onnx)

The syntax for the command is:

```shell
python inference/convert_tflite.py -c config.json --input (1, 1, 28, 28) --name jax_mnist -w saved/MNIST_Sample/../checkpoint-epoch_10.params
```

It uses the model and model args given in the `config.json` file to prepare the model

> This depends on experimental features such as [JAX-tflite conversion](https://www.tensorflow.org/lite/examples/jax_conversion/overview) and [tf2onnx](https://github.com/onnx/tensorflow-onnx),
> so making it work might take some close examinations and pedantic model poking

If ran correctly, the script creates a folder with the name provided by the `--name` parameter under the `inference` folder.
```text
inference/
│
├── <name>
       ├── model/        -   copy of model/ directory when training script was invoked
       ├── config.json   -   copy of config.json when training script was invoked
       ├── <name>.tflite - converted tflite model+weights file
       └── <name>.onnx   - converted onnx model+weights file
```
The script prints logs to the console while it runs and provides information such as test outputs of each model, and input and output node names.

JAX model is run normally, tflite with tensorflow Interpreter, onnx with onnxruntime

There's still a long way to go for this part, such as adding dynamic axes options, etc.

Now the world is your oyster, deploy your neural network into various services and projects!


## Test Envs
MNIST Training Sample and TFLITE/ONNX export tested with:
- Conf 1
  - Windows 11
  - Python 3.9.12
  - jax/jaxlib==0.3.2 from the jax-windows-builder pre-built wheels
  - CUDA 11.6, cuDNN 8.3
  - > IMPORTANT: use the option `--use-deprecated legacy-resolver` when installing the `requirements.txt` packages with pip under Windows 

- Conf 2
  - Ubuntu 20.04
  - Python 3.8.10
  - jax/jaxlib==0.3.7 from JAX official linux CUDA/cuDNN build
  - CUDA 11.6, cuDNN 8.3
  - > For some reason, I needed to set XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 or else it would produce a cuDNN conv error.
   The GPU used for Conf 2 was a mobile one with a very small VRAM.



## Acknowledgements

As a PyTorch guy, I've been utilizing the project: [pytorch-template](https://github.com/victoresque/pytorch-template) for a while, adding functionality as needed. This project stems from my experience using it!

The windows build for 'jaxlib' with CUDA and cuDNN is provided by [cloudhan](https://github.com/cloudhan) with the repo [jax-windows-builder](https://github.com/cloudhan/jax-windows-builder), this saved me a lot of hassle when working in a Windows environment!

If you have any questions about the project or would like to contribute, please feel free to submit an issue, PR, or send me a DM on Twitter(@jeehoonlerenard)