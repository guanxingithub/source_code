import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from torch.utils import dlpack
import os
import torch.distributed as dist
from tensorflow.python.dlpack.dlpack import to_dlpack, from_dlpack


# Initialize PyTorch distributed training
def init_torch_distributed():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def set_gpu_option():
    gpus = tf.config.list_physical_devices('GPU')
    try:
        tf.config.set_visible_devices(gpus, 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# Configure TensorFlow for a specific GPU rank
def configure_tf_gpu_for_rank(rank):
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[rank], 'GPU')
        print(f"Rank {rank} using GPU: {gpus[rank]}")


# Define the TensorFlow model
class TfDenseModel(tf.keras.Model):
    def __init__(self):
        super(TfDenseModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)
        return self.output_layer(x)

# Define the Pytroch model
class TorchDenseModel(nn.Module):
    def __init__(self):
        super(TorchDenseModel, self).__init__()
        self.dense1 = nn.Linear(208, 512)  # 208 input features, 512 output features
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.4)
        self.output_layer = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.dense1(x))  # First layer
        x = F.relu(self.dense2(x))  # Second layer
        x = self.dropout(x)        # Dropout for regularization
        x = F.relu(self.dense3(x))  # Third layer
        return F.softmax(self.output_layer(x), dim=-1)  # Output layer with softmax


# Main script
def main(args):
    rank, world_size = init_torch_distributed()
    if args.dlpack_mode in ["tf_to_pytorch", "pytorch_only"]:
        configure_tf_gpu_for_rank(rank)
    else:
        set_gpu_option()


    for i in range(5):
        # Step 1: Create a PyTorch Tensor
        generated_pytorch_tensor = torch.randn(204762, 208, device=f"cuda:{rank}")
        # Code below let pytorch to decide which device to pick, it doesn't impact the device name in dlpack
        # generated_pytorch_tensor = torch.randn(204762, 208, device="cuda")
        print(f"Rank {rank}: PyTorch Tensor Shape: {generated_pytorch_tensor.shape}")
        print(f"Rank {rank}: PyTorch tensor generated on {generated_pytorch_tensor.device}")

        # Forcing it to use other devices. However, it doesn't work because TF only can see its own device as gpu:0
        # due to Horovodrun and Torchrun initialize 1 gpu per process.
        with tf.device(f"/GPU:{rank}"):
            generated_tf_tensor = tf.random.uniform((204762, 208), dtype=tf.float32)
        print(f"Rank {rank}: TensorFlow Tensor Generated on: {generated_tf_tensor.device}")

        # Step 2: Export PyTorch Tensor as DLPack and import into TensorFlow
        # Map the mode to corresponding tensors and conversion functions
        tensor_mapping = {
            "pytorch_to_tf": (generated_pytorch_tensor, dlpack.to_dlpack, from_dlpack),
            "tf_only": (generated_tf_tensor, to_dlpack, from_dlpack),
            "pytorch_only": (generated_pytorch_tensor, dlpack.to_dlpack, dlpack.from_dlpack),
            "tf_to_pytorch": (generated_tf_tensor, to_dlpack, dlpack.from_dlpack)
        }

        # Get the tensor, conversion function, and result handler based on the mode
        tensor_to_convert, dlpack_convert, dlpack_from = tensor_mapping.get(args.dlpack_mode, (generated_pytorch_tensor, dlpack.to_dlpack, from_dlpack))

        # Convert the tensor using DLPack
        dlpack_capsule = dlpack_convert(tensor_to_convert)
        tensor = dlpack_from(dlpack_capsule)

        # Step 3: Build and feed TensorFlow model

        if args.dlpack_mode in ["tf_to_pytorch", "pytorch_only"]:
            model = TorchDenseModel().to(f"cuda:{rank}")
            tensor = tensor.to(f"cuda:{rank}")
            output = model(tensor)
        else:
            with tf.device(f"/GPU:{rank}"):
                model = TfDenseModel()
                output = model(tensor)

        print(f"Rank {rank}: Model Output Shape: {output.shape}")
        print(f"Rank {rank}: Model Output: {output}")

    # Destroy the process group
    dist.destroy_process_group()
    print(f"Rank {rank}: Process group destroyed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch and TensorFlow DLPack Example"
    )
    parser.add_argument(
        "--dlpack_mode", 
        type=str, 
        default="pytorch_to_tf", 
        help=(
            "Specifies how tensors should be converted using DLPack. "
            "Options are: 'pytorch_to_tf' to convert PyTorch tensors to TensorFlow tensors, "
            "'tf_to_pytorch' to convert TensorFlow tensors to PyTorch tensors, "
            "'tf_only' to test TensorFlow tensors only, or "
            "'pytorch_only' to test PyTorch tensors only. The default is 'pytorch_to_tf'."
        )
    )
    args = parser.parse_args()
    main(args)
