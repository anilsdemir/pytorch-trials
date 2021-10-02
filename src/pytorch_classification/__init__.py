import logging.handlers
import os
from pathlib import Path

from pytorch_classification.config import(
    LOCAL_DIR,
    LOGGING_DIR
)
logger = logging.getLogger(__name__)

folders_to_be_created = [
    LOCAL_DIR,
    LOGGING_DIR,
]

formatter = logging.Formatter(
    "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    "%d-%m %H:%M:%S",
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

log_file_name = Path.joinpath(LOGGING_DIR, "mnist-classification.log")

if log_file_name.is_file():
    rotating_file_handler = logging.handlers.RotatingFileHandler(
        log_file_name,
        maxBytes=10048576,  # 10 MB
        backupCount=10,
    )
    rotating_file_handler.setFormatter(formatter)

    logger.addHandler(rotating_file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(logging.DEBUG)
else:
    for folder in folders_to_be_created:
        try:
            print(folder)
            os.mkdir(folder)
        except FileExistsError:
            pass
    open(log_file_name, mode='w').close()
    logging.basicConfig(filename=log_file_name, level=logging.INFO)
