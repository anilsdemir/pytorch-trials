import logging.handlers

from pathlib import Path

from pytorch_classification.config import LOGGING_DIR

logger = logging.getLogger(__name__)

formatter = logging.Formatter(
    "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    "%d-%m %H:%M:%S",
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

log_file_name = Path.joinpath(LOGGING_DIR, "mnist-classification.log")

rotating_file_handler = logging.handlers.RotatingFileHandler(
    log_file_name,
    maxBytes=10048576,  # 10 MB
    backupCount=10,
)
rotating_file_handler.setFormatter(formatter)

logger.addHandler(rotating_file_handler)
logger.addHandler(console_handler)

logger.setLevel(logging.DEBUG)
