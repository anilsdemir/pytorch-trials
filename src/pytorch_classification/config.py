import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

LOCAL_DIR = Path.joinpath(BASE_DIR, "local")

LOGGING_DIR = Path.joinpath(LOCAL_DIR, "logs")

INPUT_DIR = Path.joinpath(LOCAL_DIR, "inputs")

OUTPUT_DIR = Path.joinpath(LOCAL_DIR, "outputs")

#print(BASE_DIR, "\n", LOCAL_DIR, "\n", LOGGING_DIR, "\n", INPUT_DIR, "\n",
      #OUTPUT_DIR)
