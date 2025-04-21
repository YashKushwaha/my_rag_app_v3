#import argparse
#from src.indexing import run_indexing_pipeline

import warnings
warnings.filterwarnings("ignore")

from src.config_loader import get_config
import os
from pathlib import Path
from src.indexing import load_indexer

if __name__ == "__main__":
    os.chdir(Path(os.path.dirname(__file__)).resolve())
    config = get_config('config/index_creation.yaml')
    indexer = load_indexer(config)
    indexer.ingest()
    indexer.index()
