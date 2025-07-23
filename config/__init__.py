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
from .curriculum_config import CurriculumConfig, StageConfig
from .generation_config import GenerationConfig

@dataclass
class ProjectConfig:
    """Unified configuration manager"""
    model: ModelConfig
    training: TrainingConfig
    curriculum: CurriculumConfig
    generation: GenerationConfig
    
    # Project paths
    data_dir: str = "data"
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    
    # Hardware settings
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Experiment settings
    experiment_name: str = "micro-diffusion"
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ProjectConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert curriculum stages from dicts to StageConfig objects
        curriculum_dict = config_dict.get('curriculum', {})
        if 'stages' in curriculum_dict:
            stage_configs = []
            for stage_dict in curriculum_dict['stages']:
                # Convert masking_rate_range list to tuple
                if 'masking_rate_range' in stage_dict:
                    stage_dict['masking_rate_range'] = tuple(stage_dict['masking_rate_range'])
                
                # Create StageConfig object
                stage_config = StageConfig(**stage_dict)
                stage_configs.append(stage_config)
            
            # Replace stages list with StageConfig objects
            curriculum_dict['stages'] = stage_configs
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            curriculum=CurriculumConfig(**curriculum_dict),
            generation=GenerationConfig(**config_dict.get('generation', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['model', 'training', 'curriculum', 'generation']}
        )
    
    @classmethod
    def default(cls) -> 'ProjectConfig':
        """Create default configuration for the Micro Diffusion project"""
        return cls(
            model=ModelConfig.tiny_125m(),
            training=TrainingConfig.default(),
            curriculum=CurriculumConfig.three_stage(),
            generation=GenerationConfig.default()
        )
    
    def save_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training), 
            'curriculum': self._curriculum_to_dict(),
            'generation': asdict(self.generation),
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
        
        # Convert StageConfig objects to dicts and tuples to lists
        if 'stages' in curriculum_dict:
            for stage in curriculum_dict['stages']:
                if 'masking_rate_range' in stage and isinstance(stage['masking_rate_range'], tuple):
                    stage['masking_rate_range'] = list(stage['masking_rate_range'])
        
        return curriculum_dict
    
    def validate(self):
        """Validate all configuration components"""
        # Validate model configuration
        assert self.model.d_model > 0, "Model dimension must be positive"
        assert self.model.n_layers > 0, "Number of layers must be positive"
        assert self.model.n_heads > 0, "Number of heads must be positive"
        assert self.model.d_model % self.model.n_heads == 0, "Model dimension must be divisible by number of heads"
        assert self.model.vocab_size > 0, "Vocabulary size must be positive"
        assert self.model.max_seq_len > 0, "Maximum sequence length must be positive"
        assert 0 <= self.model.dropout <= 1, "Dropout must be between 0 and 1"
        assert 0 <= self.model.attention_dropout <= 1, "Attention dropout must be between 0 and 1"
        
        # Validate training configuration
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert self.training.learning_rate > 0, "Learning rate must be positive"
        assert self.training.weight_decay >= 0, "Weight decay must be non-negative"
        assert self.training.warmup_steps >= 0, "Warmup steps must be non-negative"
        assert self.training.max_grad_norm > 0, "Max gradient norm must be positive"
        
        # Validate curriculum configuration
        self.curriculum.validate()
        
        # Validate generation configuration
        self.generation.validate()
        
        # Validate path configurations
        assert isinstance(self.data_dir, str), "Data directory must be a string"
        assert isinstance(self.output_dir, str), "Output directory must be a string"
        assert isinstance(self.cache_dir, str), "Cache directory must be a string"
        
        # Validate device configuration
        assert self.device in ["cuda", "cpu", "auto"], f"Invalid device: {self.device}"
        
        # Validate experiment settings
        assert isinstance(self.experiment_name, str), "Experiment name must be a string"
        assert isinstance(self.seed, int) and self.seed >= 0, "Seed must be a non-negative integer"
        
        return True

# Convenience functions
def get_default_config() -> ProjectConfig:
    """Get default project configuration"""
    return ProjectConfig.default()

def load_config(config_path: str) -> ProjectConfig:
    """Load configuration from file"""
    return ProjectConfig.from_yaml(config_path)