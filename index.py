import argparse
from src.indexing import run_indexing_pipeline
from src.config_loader import get_config
import os
from pathlib import Path

if __name__ == "__main__":
    os.chdir(Path(os.path.dirname(__file__)).resolve())
    config = get_config()
    run_indexing_pipeline(config)
