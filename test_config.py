#!/usr/bin/env python3
"""Test the configuration system"""

import sys
sys.path.append('.')

from config import ProjectConfig

def test_config():
    """Test configuration loading and saving"""
    
    # Create default config
    config = ProjectConfig.default()
    
    # Validate all components
    config.model.validate()
    config.training.validate() 
    config.curriculum.validate()
    
    # Print summary
    print("🔧 Configuration Summary:")
    print(f"Model: {config.model.param_count:,} parameters")
    print(f"Stages: {len(config.curriculum.stages)} ({config.curriculum.total_epochs} total epochs)")
    print(f"Batch size: {config.training.effective_batch_size}")
    print(f"Memory: {'✅ Optimized' if config.gradient_checkpointing else '❌ Not optimized'}")
    
    # Save config for later use
    config.save_yaml("config.yaml")
    print("✅ Configuration saved to config.yaml")
    
    # Test loading
    loaded_config = ProjectConfig.from_yaml("config.yaml") 
    assert loaded_config.model.d_model == config.model.d_model
    print("✅ Configuration loading works")

if __name__ == "__main__":
    test_config()