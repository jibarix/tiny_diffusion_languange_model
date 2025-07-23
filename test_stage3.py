#!/usr/bin/env python3
"""
Direct Stage 3 Test - Skip to paragraph format training

Tests the RoPE fix by jumping directly to Stage 3 (refinement) 
which uses longer sequences in paragraph format.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import ProjectConfig
from src.data import DataPipeline
from src.trainer import create_trainer_from_config

def test_stage3_directly():
    """Test Stage 3 directly with paragraph format"""
    print("Testing Stage 3 (refinement) directly...")
    
    # Create debug config
    config = ProjectConfig.debug()
    config_dict = config.to_dict()
    
    # Process data
    data_pipeline = DataPipeline(config_dict)
    data_pipeline.process_book("data/raw/frankenstein.txt", save_dir="data/processed")
    
    # Create trainer
    trainer = create_trainer_from_config(config_dict, data_pipeline, device='cuda')
    
    # Skip directly to Stage 3 (index 2)
    print("Jumping directly to Stage 3...")
    try:
        stage_results = trainer.train_stage(2)  # Stage 3 (refinement)
        print(f"[OK] Stage 3 completed successfully!")
        print(f"Final loss: {stage_results.final_loss:.4f}")
        return True
    except Exception as e:
        print(f"[FAILED] Stage 3 failed: {e}")
        return False

if __name__ == "__main__":
    success = test_stage3_directly()
    sys.exit(0 if success else 1)