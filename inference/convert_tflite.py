import argparse
import json
import os
import pickle
import sys
from functools import partial
from pathlib import Path
from shutil import copytree, ignore_patterns

import jax.numpy as jnp
import numpy as np
import onnxruntime
import tensorflow as tf
import tf2onnx

sys.path.append('./')
import model as model_module
from utils import get_training_objects


def main(cfg):
    # Create folder under the inference folder
    path = Path("inference", cfg.name)
    if path.exists():
        print("Name already used, please choose a different name!")
        # return -1
    else:
        os.mkdir(path)

    # Configuration
    with open(Path(cfg.config), 'r') as f:
        config = json.load(f)

    # Parse Arguments to create Model instance with pretrained weights
    config['model_struct']['args'] = {k: tuple(v) if isinstance(v, list) else v for k, v in
                                      config['model_struct']['args'].items()}
    model = get_training_objects(model_module, config['model_struct']['model'], "model")
    with open(Path(cfg.ckpt), 'rb') as f:
        pretrained_weights = pickle.load(f)
    model = partial(model, **config['model_struct']['args'], params=pretrained_weights)

    # Dummy input for testing
    dummy_input = jnp.zeros(cfg.input).astype("float32")
    numpy_input = np.asarray(dummy_input)

    # TFLite Converter - convert JAX xla representation into tflite bytes and save to folder
    converter = tf.lite.TFLiteConverter.experimental_from_jax([model], [[('input', dummy_input)]])
    tflite_model = converter.convert()
    with open(Path(path, f"{cfg.name}.tflite"), 'wb') as f:
        f.write(tflite_model)

    # Test converted result
    expected, w = model(dummy_input)
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]["index"])
    np.testing.assert_almost_equal(expected, result, 1e-5)
    print(f"\nExpected: {expected}")
    print(f"TFLite: {result}\n")
    print("TFLite model conversion successful!\n")

    # ONNX conversion
    try:
        model_proto, _ = tf2onnx.convert._convert_common(None, tflite_path=str(Path(path, f"{cfg.name}.tflite")), opset=15,
                                                         output_path=str(Path(path, f"{cfg.name}.onnx")))
        print("Model inputs: ", [n.name for n in model_proto.graph.input])
        print("Model outputs:", [n.name for n in model_proto.graph.output])

        # Test result with onnxruntime
        ort_session = onnxruntime.InferenceSession(str(Path(path, f"{cfg.name}.onnx")), providers=["CPUExecutionProvider"])

        ort_input = {ort_session.get_inputs()[0].name: numpy_input}
        ort_outs = ort_session.run(None, ort_input)
        print(f"\nONNX: {ort_outs[0]}\n")
        np.testing.assert_almost_equal(expected, ort_outs[0], 1e-5)
        print("Converted ONNX model is good!\n")
    except Exception:
        print("ONNX Conversion error!, skipping onnx model saving")
        if Path(path, f"{cfg.name}.onnx").exists():
            os.remove(Path(path, f"{cfg.name}.onnx"))
        return -1

    # Copy train configuration to save_dir
    with open(Path(path, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    # Copy model folder to save_dir
    copytree(Path("./model"), Path(path, "model"), ignore=ignore_patterns("__init__.py"))


if __name__ == "__main__":
    # Command line argument parser
    parser = argparse.ArgumentParser(description="tflite Converter")
    parser.add_argument('-c', '--config', help='Model Configuration', required=True)
    parser.add_argument('-w', '--ckpt', help='Model Checkpoint', required=True)
    parser.add_argument('-i', '--input', nargs="+", help='Input Size', type=int, required=True)
    parser.add_argument('-n', '--name', help="Inference package name", required=True)

    args = parser.parse_args()
    main(args)

