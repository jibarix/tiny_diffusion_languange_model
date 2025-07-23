#!/usr/bin/env python3
"""
Debug CUDA Error - Find exact source
"""

import os
import sys
from pathlib import Path

# Enable CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import ProjectConfig
from src.data import DataPipeline
from src.trainer import create_trainer_from_config

def debug_stage3():
    """Debug Stage 3 with CUDA blocking enabled"""
    print("Debugging Stage 3 with CUDA_LAUNCH_BLOCKING=1...")
    
    config = ProjectConfig.debug()
    config_dict = config.to_dict()
    
    # Process data
    data_pipeline = DataPipeline(config_dict)
    data_pipeline.process_book("data/raw/frankenstein.txt", save_dir="data/processed")
    
    # Create trainer
    trainer = create_trainer_from_config(config_dict, data_pipeline, device='cuda')
    
    # Test Stage 3
    try:
        print("Starting Stage 3 with detailed error tracking...")
        stage_results = trainer.train_stage(2)
        print("[OK] Success!")
        return True
    except Exception as e:
        print(f"[FAILED] Detailed error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_stage3()