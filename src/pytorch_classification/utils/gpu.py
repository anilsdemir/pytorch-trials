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
    logging.warning("CUDA device created")

    if device.type == 'cuda':
        logging.warning(
            '<--- Memory Usage --->'
        )
        logging.warning(
            'Allocated: '
            f'{str(round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1))} GB'
        )
        logging.warning(
            'Cached: '
            f'{str(round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1))} GB'
        )

    return device


def get_gpu_info():
    if not torch.cuda.is_available():
        raise CudaIsNotAvailable(
            f"No available cudas were found at this time."
            f"Please close programs that use cuda."
        )
    logging.info(
        '"""""""""""""""""""""""""""""\n'
        f"Device Name: {torch.cuda.get_device_name(0)}\n"
        f"Device Count: {torch.cuda.device_count()}\n"
        f"Device Capability: {torch.cuda.get_device_capability()}\n"
        '"""""""""""""""""""""""""""""'
    )
