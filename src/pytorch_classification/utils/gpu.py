import logging
from pathlib import Path

import torch

from pytorch_classification.config import (
    LOCAL_DIR,
    LOGGING_DIR
)

from pytorch_classification.utils.exceptions import (
    CudaIsNotAvailable
)


def create_gpu_device():
    if not torch.cuda.is_available():
        raise CudaIsNotAvailable(
            f"No available cudas were found at this time."
            f"Please close programs that use cuda."
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def get_gpu_info():
    if not torch.cuda.is_available():
        raise CudaIsNotAvailable(
            f"No available cudas were found at this time."
            f"Please close programs that use cuda."
        )
    print(
        f"Device Name: {torch.cuda.get_device_name(0)}\n"
        f"Device Count: {torch.cuda.device_count()}\n"
        f"Device Capability: {torch.cuda.get_device_capability()}"
    )

