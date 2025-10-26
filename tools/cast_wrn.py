# -*- coding: utf-8 -*-
"""# A· Wide Residual Network"""

import torch
from freeze_torch_model import MarsSmall128
import os
import tensorflow as tf
import numpy as np

def extract_constants_from_graph(pb_file, output_dir="tf_constants"):
    """
    The model will be saved in a tree directory strcuter under `tf_constants` directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(pb_file, "rb") as f:
        graph_def.ParseFromString(f.read())

    for node in graph_def.node:
        if node.op == "Const":  # constant node, usually contains weights
            tensor_value = tf.make_ndarray(node.attr["value"].tensor)
            dirname = os.path.dirname(node.name)
            os.makedirs(os.path.join(output_dir, dirname), exist_ok=True)
            np.save(os.path.join(output_dir, node.name + ".npy"), tensor_value)
            print(f"Saved constant node: {node.name} shape {tensor_value.shape}")

def map_tf_to_pt(tf_name):
    """
    Maps extracted TF constant filename -pytorch parameter name
    """

    # Skip classifier
    if "ball" in tf_name:
        return None

    mapping = {
        # --- Input convs ---
        "conv1_1/weights.npy": "conv1.weight",
        "conv1_1/conv1_1/bn/Const.npy": "conv1_bn.weight",
        "conv1_1/conv1_1/bn/beta.npy": "conv1_bn.bias",
        "conv1_1/conv1_1/bn/moving_mean.npy": "conv1_bn.running_mean",
        "conv1_1/conv1_1/bn/moving_variance.npy": "conv1_bn.running_var",

        "conv1_2/weights.npy": "conv2.weight",
        "conv1_2/conv1_2/bn/Const.npy": "conv2_bn.weight",
        "conv1_2/conv1_2/bn/beta.npy": "conv2_bn.bias",
        "conv1_2/conv1_2/bn/moving_mean.npy": "conv2_bn.running_mean",
        "conv1_2/conv1_2/bn/moving_variance.npy": "conv2_bn.running_var",

        # Res1
        "conv2_1/1/weights.npy"                     : "res1.conv1.conv.weight",
        "conv2_1/1/conv2_1/1/bn/Const.npy"          : "res1.bn1.weight",
        "conv2_1/1/conv2_1/1/bn/beta.npy"           : "res1.bn1.bias",
        "conv2_1/1/conv2_1/1/bn/moving_mean.npy"    : "res1.bn1.running_mean",
        "conv2_1/1/conv2_1/1/bn/moving_variance.npy": "res1.bn1.running_var",
        "conv2_1/2/weights.npy"                     : "res1.conv2.weight",
        "conv2_1/2/biases.npy"                      : "res1.conv2.bias",

        # Res2
        "conv2_3/bn/Const.npy"                      : "res2.pre_bn.weight",
        "conv2_3/bn/beta.npy"                       : "res2.pre_bn.bias",
        "conv2_3/bn/moving_mean.npy"                : "res2.pre_bn.running_mean",
        "conv2_3/bn/moving_variance.npy"            : "res2.pre_bn.running_var",
        "conv2_3/1/weights.npy"                     : "res2.conv1.conv.weight",
        "conv2_3/1/conv2_3/1/bn/Const.npy"          : "res2.bn1.weight",
        "conv2_3/1/conv2_3/1/bn/beta.npy"           : "res2.bn1.bias",
        "conv2_3/1/conv2_3/1/bn/moving_mean.npy"    : "res2.bn1.running_mean",
        "conv2_3/1/conv2_3/1/bn/moving_variance.npy": "res2.bn1.running_var",
        "conv2_3/2/weights.npy"                     : "res2.conv2.weight",
        "conv2_3/2/biases.npy"                      : "res2.conv2.bias",

        # Res3
        "conv3_1/bn/Const.npy"                      : "res3.pre_bn.weight",
        "conv3_1/bn/beta.npy"                       : "res3.pre_bn.bias",
        "conv3_1/bn/moving_mean.npy"                : "res3.pre_bn.running_mean",
        "conv3_1/bn/moving_variance.npy"            : "res3.pre_bn.running_var",
        "conv3_1/1/weights.npy"                     : "res3.conv1.conv.weight",
        "conv3_1/1/conv3_1/1/bn/Const.npy"          : "res3.bn1.weight",
        "conv3_1/1/conv3_1/1/bn/beta.npy"           : "res3.bn1.bias",
        "conv3_1/1/conv3_1/1/bn/moving_mean.npy"    : "res3.bn1.running_mean",
        "conv3_1/1/conv3_1/1/bn/moving_variance.npy": "res3.bn1.running_var",
        "conv3_1/2/weights.npy"                     : "res3.conv2.weight",
        "conv3_1/2/biases.npy"                      : "res3.conv2.bias",
        "conv3_1/projection/weights.npy"            : "res3.downsample.weight",

        # Res4
        "conv3_3/bn/Const.npy"                      : "res4.pre_bn.weight",
        "conv3_3/bn/beta.npy"                       : "res4.pre_bn.bias",
        "conv3_3/bn/moving_mean.npy"                : "res4.pre_bn.running_mean",
        "conv3_3/bn/moving_variance.npy"            : "res4.pre_bn.running_var",
        "conv3_3/1/weights.npy"                     : "res4.conv1.conv.weight",
        "conv3_3/1/conv3_3/1/bn/Const.npy"          : "res4.bn1.weight",
        "conv3_3/1/conv3_3/1/bn/beta.npy"           : "res4.bn1.bias",
        "conv3_3/1/conv3_3/1/bn/moving_mean.npy"    : "res4.bn1.running_mean",
        "conv3_3/1/conv3_3/1/bn/moving_variance.npy": "res4.bn1.running_var",
        "conv3_3/2/weights.npy"                     : "res4.conv2.weight",
        "conv3_3/2/biases.npy"                      : "res4.conv2.bias",
        "conv3_3/projection/weights.npy"            : "res4.downsample.weight",

        # Res5
        "conv4_1/bn/Const.npy"                      : "res5.pre_bn.weight",
        "conv4_1/bn/beta.npy"                       : "res5.pre_bn.bias",
        "conv4_1/bn/moving_mean.npy"                : "res5.pre_bn.running_mean",
        "conv4_1/bn/moving_variance.npy"            : "res5.pre_bn.running_var",
        "conv4_1/1/weights.npy"                     : "res5.conv1.conv.weight",
        "conv4_1/1/conv4_1/1/bn/Const.npy"          : "res5.bn1.weight",
        "conv4_1/1/conv4_1/1/bn/beta.npy"           : "res5.bn1.bias",
        "conv4_1/1/conv4_1/1/bn/moving_mean.npy"    : "res5.bn1.running_mean",
        "conv4_1/1/conv4_1/1/bn/moving_variance.npy": "res5.bn1.running_var",
        "conv4_1/2/weights.npy"                     : "res5.conv2.weight",
        "conv4_1/2/biases.npy"                      : "res5.conv2.bias",
        "conv4_1/projection/weights.npy"            : "res5.downsample.weight",

        # Res6
        "conv4_3/bn/Const.npy"                      : "res6.pre_bn.weight",
        "conv4_3/bn/beta.npy"                       : "res6.pre_bn.bias",
        "conv4_3/bn/moving_mean.npy"                : "res6.pre_bn.running_mean",
        "conv4_3/bn/moving_variance.npy"            : "res6.pre_bn.running_var",
        "conv4_3/1/weights.npy"                     : "res6.conv1.conv.weight",
        "conv4_3/1/conv4_3/1/bn/Const.npy"          : "res6.bn1.weight",
        "conv4_3/1/conv4_3/1/bn/beta.npy"           : "res6.bn1.bias",
        "conv4_3/1/conv4_3/1/bn/moving_mean.npy"    : "res6.bn1.running_mean",
        "conv4_3/1/conv4_3/1/bn/moving_variance.npy": "res6.bn1.running_var",
        "conv4_3/2/weights.npy"                     : "res6.conv2.weight",
        "conv4_3/2/biases.npy"                      : "res6.conv2.bias",

        # --- Final FC + BN ---
        "fc1/weights.npy": "fc.weight",
        "fc1/biases.npy": "fc.bias",
        "fc1/fc1/bn/Const.npy": "bn.weight",
        "fc1/fc1/bn/beta.npy": "bn.bias",
        "fc1/fc1/bn/moving_mean.npy": "bn.running_mean",
        "fc1/fc1/bn/moving_variance.npy": "bn.running_var",

        "fc1/fc1/bn/Reshape/shape.npy": None,
        "map/Const.npy" : None,
        "map/while/add/y.npy" : None,
        "map/while/strided_slice/stack_1.npy" : None,
        "map/while/strided_slice/stack_2.npy" : None,
        "map/while/strided_slice/stack.npy" : None,
        "map/TensorArrayStack/range/delta.npy": None,
        "map/TensorArrayStack/range/start.npy": None,
        "map/TensorArrayUnstack/range/delta.npy": None,
        "map/TensorArrayUnstack/range/start.npy": None,
        "map/TensorArrayUnstack/strided_slice/stack_2.npy": None,
        "map/TensorArrayUnstack/strided_slice/stack_1.npy": None,
        "map/TensorArrayUnstack/strided_slice/stack.npy": None,
        "map/strided_slice/stack_2.npy" : None,
        "map/strided_slice/stack_1.npy" : None,
        "map/strided_slice/stack.npy" : None,
        "Sum/reduction_indices.npy": None,
        "Flatten/flatten/Reshape/shape/1.npy" : None,
        "Flatten/flatten/strided_slice/stack_2.npy" : None,
        "Flatten/flatten/strided_slice/stack_1.npy" : None,
        "Flatten/flatten/strided_slice/stack.npy" : None,
        "Const.npy" : None,
    }

    return mapping[tf_name]

def load_tf_constangs_into_mars(tf_constants_dir, model):
    """
    Recursively load TF constants from nested directories into PyTorch model.

    tf_constants_dir: root folder containing TF .npy constants (nested)
    model: PyTorch model
    map_tf_to_pt: function(tf_name:str) -> pt_name:str or None
    """
    state_dict = model.state_dict()
    loaded_count = 0

    for root, dirs, files in os.walk(tf_constants_dir):
        for fname in files:
            print("\n   ", fname)
            if not fname.endswith(".npy"):
                continue

            abs_path = os.path.join(root, fname)
            # normalize path to TF variable name (like extractor did)
            tf_name = os.path.relpath(abs_path, tf_constants_dir)

            pt_name = map_tf_to_pt(tf_name)
            if pt_name is None:
                continue
            if pt_name not in state_dict:
                print(f"[-----------SKIP-0] {pt_name} not in state_dict")
                continue

            arr = np.load(abs_path)
            # permute conv/dense if needed
            if arr.ndim == 4:  # conv
                arr = arr.transpose(3, 2, 0, 1).copy()
            elif arr.ndim == 2:  # dense
                arr = arr.T.copy()

            tensor = torch.from_numpy(arr)
            if tensor.shape != state_dict[pt_name].shape:
                print(f"[-----------SKIP-1] Shape mismatch {tf_name} -> {pt_name} {tensor.shape} vs {state_dict[pt_name].shape}")
                continue

            state_dict[pt_name].copy_(tensor)
            loaded_count += 1
            print(f"[LOADED] {tf_name} -> {pt_name} {tensor.shape}")

    model.load_state_dict(state_dict)
    print(f"Done loading TF weights. Total loaded: {loaded_count}")



def save_metrics(filename, epoch, average_loss, mAP, cmc, lr):
    fd = open(filename, 'a')
    fd.write(f"{epoch + 1} {average_loss:.4f} {mAP:.4f} {cmc[0]:.4f} {cmc[4]:.4f} {cmc[9]:.4f} {lr:.6f}\n")
    fd.close()

def mk_metrics(filename):
    fd = open(filename, "w")
    fd.write("# Epoch Loss mAP Rank-1 Rank-5 Rank-10 lr\n")
    fd.close()

def save_as_pytorch(model, path="marsPytorch-custom.pth"):
    model_cpu = model.to("cpu")
    torch.save({"model_state_dict": model_cpu.state_dict()}, path)
    print(f"Saved pytorch checkpoint to {path}")

import argparse
def user_config():
    parser = argparse.ArgumentParser("Cast WRN from TensorFlowV1 to pytorch.")

    parser.add_argument("--dir",
                       help="Directory where the tensor flow v1 model was disclosed.",
                       default="/home/chris/Documents/Code/mot/experiments/trackers/deep_sort/tools/tf_constants")
    parser.add_argument("--filename",
                        help="in tf2dir: this means the model to upload and in dir2pth: this means the pytorch model",
                        default="mars-small128-pytorch.pth")

    parser.add_argument("--cast",
                        help="Can either be tf2dir: unfolds the tensorflow graph into a directory tree, or dir2torch: grabs the directory tree tensor flow graph into a pytorch model."
                        type=str,
                        default="tf2dir")

    return parser.parse_args()

if "__main__" == __name__:

    args = user_config()
    
    if "tf2dir" == args.cast:
        extract_constants_from_graph(args.filename)

    elif "dir2torch" == args.cast:
        model = MarsSmall128(num_classes=1).cuda()
        load_tf_constangs_into_mars(args.dir, model)
        save_as_pytorch(model, path=args.new_name)
