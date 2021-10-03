import os

import torch
import numpy as np


def _create_tensor_examples():
    x = torch.rand(5, 3)
    print(x)

    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(x_data)

    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print(x_np)

    x_ones = torch.ones_like(x_data)  # retains the properties of x_data
    print(f"Ones Tensor: \n {x_ones} \n")

    x_rand = torch.rand_like(
        x_data,
        dtype=torch.float
    )  # overrides the datatype of x_data
    print(f"Random Tensor: \n {x_rand} \n")

    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")

    tensor = torch.rand(3, 4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

    # By default, tensors are created on the CPU. We need to explicitly move
    # tensors to the GPU using .to method

    if torch.cuda.is_available():

        tensor = torch.ones(4, 4)
        print('First row: ', tensor[0])
        print('First column: ', tensor[:, 0])
        print('Last column:', tensor[..., -1])
        tensor = tensor.to("cuda")
        tensor[:, 1] = 0
        print(tensor)


if __name__ == '__main__':
    _create_tensor_examples()
