import os
import logging
from pathlib import Path

import torch
import numpy as np

from pytorch_classification.utils.gpu import (
    get_gpu_info
)


if __name__ == '__main__':
    get_gpu_info()
