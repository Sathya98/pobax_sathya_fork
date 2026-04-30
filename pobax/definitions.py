import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = str(Path(ROOT_DIR).parent)
IVI_STORAGE_DIR = '/ivi/zfs/s0/original_homes/sbhetha/urnn_results'