"""
Unified Configuration Manager for Tiny Text Diffusion Model
Provides single entry point for all configurations
"""

import os
from dataclasses import dataclass, asdict
from typing import Dict, Any
import yaml

from .model_config import ModelConfig
from .training_config import TrainingConfig  
from .curriculum_config import CurriculumConfig


@dataclass
class ProjectConfig:
    """Unified configuration manager"""
    model: ModelConfig
    training: TrainingConfig
    curriculum: CurriculumConfig
    
    # Project paths
    data_dir: str = "data"
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    
    # Hardware settings
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Experiment settings
    experiment_name: str = "darwin-diffusion"
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ProjectConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert curriculum lists back to tuples
        curriculum_dict = config_dict.get('curriculum', {})
        if 'stages' in curriculum_dict:
            for stage in curriculum_dict['stages']:
                if 'masking_rate_range' in stage:
                    stage['masking_rate_range'] = tuple(stage['masking_rate_range'])
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            curriculum=CurriculumConfig(**curriculum_dict),
            **{k: v for k, v in config_dict.items() 
               if k not in ['model', 'training', 'curriculum']}
        )
    
    @classmethod
    def default(cls) -> 'ProjectConfig':
        """Create default configuration for the Darwin project"""
        return cls(
            model=ModelConfig.tiny_125m(),
            training=TrainingConfig.default(),
            curriculum=CurriculumConfig.three_stage()
        )
    
    def save_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training), 
            'curriculum': self._curriculum_to_dict(),
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'cache_dir': self.cache_dir,
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            'gradient_checkpointing': self.gradient_checkpointing,
            'experiment_name': self.experiment_name,
            'seed': self.seed
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, indent=2)
    
    def _curriculum_to_dict(self):
        """Convert curriculum config to YAML-safe dict"""
        curriculum_dict = asdict(self.curriculum)
        # Convert tuples to lists for YAML serialization
        for stage in curriculum_dict['stages']:
            stage['masking_rate_range'] = list(stage['masking_rate_range'])
        return curriculum_dict
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'curriculum': asdict(self.curriculum),
            'data_dir': self.data_dir,
            'output_dir': self.output_dir, 
            'cache_dir': self.cache_dir,
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            'gradient_checkpointing': self.gradient_checkpointing,
            'experiment_name': self.experiment_name,
            'seed': self.seed
        }


# Convenience functions
def get_default_config() -> ProjectConfig:
    """Get default project configuration"""
    return ProjectConfig.default()

def load_config(config_path: str) -> ProjectConfig:
    """Load configuration from file"""
    return ProjectConfig.from_yaml(config_path)