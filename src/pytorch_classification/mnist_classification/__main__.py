import os
import logging
from pathlib import Path

import torch
import numpy as np

from pytorch_classification.utils.gpu import (
    get_gpu_info,
    create_gpu_device
)


if __name__ == '__main__':
    device = create_gpu_device()
    get_gpu_info()

