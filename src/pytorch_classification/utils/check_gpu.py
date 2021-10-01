import logging

import torch

print(torch.cuda.get_device_name(0))

logger = logging.getLogger(__name__)

logger.warning('I am warning')

logger.setLevel(logging.INFO)