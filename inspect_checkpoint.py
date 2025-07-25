#!/usr/bin/env python3
"""
Inspect the exact contents of the checkpoint file
"""

import torch
import json

def inspect_checkpoint():
    checkpoint_path = "outputs/checkpoints/best_stage2.pt"
    
    print("=== CHECKPOINT INSPECTION ===")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("Checkpoint keys:")
    for key in checkpoint.keys():
        if key != 'model_state_dict':  # Skip the huge model weights
            print(f"  {key}: {checkpoint[key]}")
    
    print(f"\nStage-related keys:")
    print(f"  current_stage_idx: {checkpoint.get('current_stage_idx', 'MISSING')}")
    print(f"  completed_stages: {checkpoint.get('completed_stages', 'MISSING')}")
    print(f"  stage_results: {checkpoint.get('stage_results', 'MISSING')}")
    
    # Check if these exist and their types
    if 'current_stage_idx' in checkpoint:
        value = checkpoint['current_stage_idx']
        print(f"  current_stage_idx type: {type(value)}, value: {value}")
    
    if 'completed_stages' in checkpoint:
        value = checkpoint['completed_stages']
        print(f"  completed_stages type: {type(value)}, value: {value}")

if __name__ == "__main__":
    inspect_checkpoint()