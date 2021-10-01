"""
Script to be used to create local folders for running the project successfully.

You need to run this script only once when you fresh clone the repo.
If the script changes you might need to run it again to create new directories.

Make sure that your OUTPUT_SOURCE is set to LOCAL, so the output directories
can be created with correct paths.
"""

import os
import sys

from pytorch_classification.config import (
    BASE_DIR,
    LOCAL_DIR,
    INPUT_DIR,
    LOGGING_DIR,
    OUTPUT_DIR,
)

folders_to_be_created = [
    BASE_DIR,
    LOCAL_DIR,
    INPUT_DIR,
    LOGGING_DIR,
    OUTPUT_DIR,
]

if __name__ == "__main__":
    with open(log_folder, 'w') as fp:
        pass
    for folder in folders_to_be_created:
        try:
            os.mkdir(folder)
            logger.info(f"Created folder with path: {folder}")
        except FileExistsError:
            pass
