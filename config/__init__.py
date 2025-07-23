"""
Simplified Configuration Manager for Tiny Text Diffusion Model - UPDATED VERSION
Single source of truth with simple override system - NO YAML COMPLEXITY
"""

import os
import argparse
from dataclasses import dataclass, replace, fields
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .model_config import ModelConfig
from .training_config import TrainingConfig  
from .curriculum_config import CurriculumConfig
from .generation_config import GenerationConfig
from .pipeline_config import PipelineConfig
from .evaluation_config import EvaluationConfig  # NEW: Added evaluation config


@dataclass
class ProjectConfig:
    """Simplified configuration manager - Python is the source of truth - UPDATED VERSION"""
    model: ModelConfig
    training: TrainingConfig
    curriculum: CurriculumConfig
    generation: GenerationConfig
    pipeline: PipelineConfig
    evaluation: EvaluationConfig  # NEW: Added evaluation configuration
    
    # Project paths
    data_dir: str = "data"
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    
    # Hardware settings
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Experiment settings
    experiment_name: str = "tiny-diffusion"
    seed: int = 42
    
    @classmethod
    def default(cls) -> 'ProjectConfig':
        """Create default configuration - SINGLE SOURCE OF TRUTH - UPDATED"""
        return cls(
            model=ModelConfig.tiny_125m(),
            training=TrainingConfig.default(),
            curriculum=CurriculumConfig.three_stage(),
            generation=GenerationConfig.default(),
            pipeline=PipelineConfig.default(),
            evaluation=EvaluationConfig.default()  # NEW: Added evaluation config
        )
    
    @classmethod
    def debug(cls) -> 'ProjectConfig':
        """Fast debug configuration - UPDATED"""
        return cls(
            model=ModelConfig.tiny_125m(),
            training=TrainingConfig.fast_debug(),
            curriculum=CurriculumConfig.fast_debug(),
            generation=GenerationConfig.fast(),
            pipeline=PipelineConfig.fast_processing(),
            evaluation=EvaluationConfig.fast(),  # NEW: Added fast evaluation config
            experiment_name="debug-run"
        )
    
    @classmethod
    def comprehensive(cls) -> 'ProjectConfig':
        """Comprehensive configuration for research - NEW"""
        return cls(
            model=ModelConfig.tiny_125m(),
            training=TrainingConfig.default(),
            curriculum=CurriculumConfig.research_config(),
            generation=GenerationConfig.creative(),
            pipeline=PipelineConfig.high_quality(),
            evaluation=EvaluationConfig.comprehensive(),
            experiment_name="comprehensive-research"
        )
    
    @classmethod
    def memory_efficient(cls) -> 'ProjectConfig':
        """Memory-efficient configuration for limited hardware - NEW"""
        return cls(
            model=ModelConfig.memory_efficient_125m(),
            training=TrainingConfig.memory_efficient(),
            curriculum=CurriculumConfig.fast_debug(),
            generation=GenerationConfig.fast(),
            pipeline=PipelineConfig.fast_processing(),
            evaluation=EvaluationConfig.memory_efficient(),
            experiment_name="memory-efficient"
        )
    
    def override(self, **kwargs) -> 'ProjectConfig':
        """Simple override with dot notation support
        
        Examples:
            config.override(seed=123, experiment_name="new-exp")
            config.override(**{"model.d_model": 512, "training.batch_size": 16})
        """
        # Handle direct field overrides
        direct_overrides = {}
        nested_overrides = {}
        
        for key, value in kwargs.items():
            if '.' in key:
                # Nested override like "model.d_model"
                section, field = key.split('.', 1)
                if section not in nested_overrides:
                    nested_overrides[section] = {}
                nested_overrides[section][field] = value
            else:
                # Direct field override
                direct_overrides[key] = value
        
        # Start with current config
        new_config = self
        
        # Apply nested overrides
        for section_name, section_overrides in nested_overrides.items():
            if hasattr(new_config, section_name):
                current_section = getattr(new_config, section_name)
                new_section = replace(current_section, **section_overrides)
                direct_overrides[section_name] = new_section
        
        # Apply all overrides
        return replace(new_config, **direct_overrides)
    
    def override_from_dict(self, override_dict: Dict[str, Any]) -> 'ProjectConfig':
        """Override from dictionary with dot notation support"""
        return self.override(**override_dict)
    
    def override_from_args(self, args: argparse.Namespace) -> 'ProjectConfig':
        """Override from command line arguments"""
        overrides = {}
        
        # Map common argument names to config paths
        arg_mapping = {
            'batch_size': 'training.batch_size',
            'learning_rate': 'training.learning_rate',
            'max_epochs': 'training.max_epochs',
            'vocab_size': 'model.vocab_size',
            'max_seq_len': 'model.max_seq_len',
            'n_layers': 'model.n_layers',
            'd_model': 'model.d_model',
            'experiment_name': 'experiment_name',
            'data_dir': 'data_dir',
            'output_dir': 'output_dir',
            'device': 'device',
            'seed': 'seed',
            # NEW: Evaluation arguments
            'eval_batch_size': 'evaluation.batch_size',
            'eval_samples': 'evaluation.max_samples',
            'generation_samples': 'evaluation.generation_num_samples'
        }
        
        # Apply mapped arguments
        for arg_name, config_path in arg_mapping.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    overrides[config_path] = value
        
        # Apply any additional arguments that match config structure
        for key, value in vars(args).items():
            if value is not None and key not in arg_mapping:
                # Try direct mapping for unmapped arguments
                if hasattr(self, key):
                    overrides[key] = value
        
        return self.override_from_dict(overrides)
    
    def override_from_file(self, config_file: Union[str, Path]) -> 'ProjectConfig':
        """Override from simple Python config file (NOT YAML!)
        
        Config file should be Python with overrides dict:
        # experiment.py
        overrides = {
            "experiment_name": "frankenstein-experiment",
            "model.vocab_size": 4196,
            "training.batch_size": 16,
            "training.learning_rate": 1e-4,
            "evaluation.batch_size": 4,
            "evaluation.max_samples": 50
        }
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Execute Python config file
        config_globals = {}
        with open(config_path, 'r') as f:
            exec(f.read(), config_globals)
        
        if 'overrides' not in config_globals:
            raise ValueError(f"Config file {config_path} must define an 'overrides' dictionary")
        
        return self.override_from_dict(config_globals['overrides'])
    
    def get_nested_value(self, path: str) -> Any:
        """Get nested config value using dot notation"""
        parts = path.split('.')
        value = self
        
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                raise AttributeError(f"Config path '{path}' not found")
        
        return value
    
    def validate(self) -> bool:
        """Validate all configuration components - UPDATED"""
        try:
            # Validate each section
            self.model.validate()
            self.training.validate(self.model)
            self.curriculum.validate()
            self.generation.validate()
            self.pipeline.validate()  # NEW: Validate pipeline config
            self.evaluation.validate()  # NEW: Validate evaluation config
            
            # Validate paths
            assert isinstance(self.data_dir, str), "data_dir must be string"
            assert isinstance(self.output_dir, str), "output_dir must be string"
            assert isinstance(self.cache_dir, str), "cache_dir must be string"
            
            # Validate device
            assert self.device in ["cuda", "cpu", "auto"], f"Invalid device: {self.device}"
            
            # Validate experiment settings
            assert isinstance(self.experiment_name, str), "experiment_name must be string"
            assert isinstance(self.seed, int) and self.seed >= 0, "seed must be non-negative int"
            
            # Cross-validation between sections
            if self.model.vocab_size != self.training.effective_vocab_size if hasattr(self.training, 'effective_vocab_size') else True:
                print("Warning: Model vocab size may not match training expectations")
            
            # NEW: Cross-validate evaluation settings with model/training
            if self.evaluation.generation_max_length > self.model.max_seq_len:
                print(f"Warning: Evaluation max length ({self.evaluation.generation_max_length}) "
                      f"exceeds model max seq len ({self.model.max_seq_len})")
            
            return True
            
        except Exception as e:
            print(f"Config validation failed: {e}")
            return False
    
    def summary(self) -> str:
        """Generate human-readable config summary - UPDATED"""
        lines = [
            f"Experiment: {self.experiment_name}",
            f"Device: {self.device}",
            f"Seed: {self.seed}",
            "",
            "Model:",
            f"  Parameters: ~{self.model.param_count:,}",
            f"  Architecture: {self.model.n_layers}L×{self.model.d_model}H×{self.model.n_heads}A",
            f"  Vocabulary: {self.model.vocab_size:,} tokens",
            f"  Sequence Length: {self.model.max_seq_len}",
            "",
            "Training:",
            f"  Batch Size: {self.training.batch_size} (effective: {self.training.effective_batch_size})",
            f"  Learning Rate: {self.training.learning_rate:.2e}",
            f"  Max Epochs: {self.training.max_epochs}",
            f"  Mixed Precision: {self.training.use_mixed_precision}",
            "",
            "Curriculum:",
            f"  Stages: {len(self.curriculum.stages)}",
            f"  Total Epochs: {self.curriculum.total_epochs}",
            "",
            "Generation:",
            f"  Temperature: {self.generation.temperature}",
            f"  Steps: {self.generation.num_steps}",
            f"  Max Length: {self.generation.max_length}",
            "",
            # NEW: Evaluation summary
            "Evaluation:",
            f"  Test Samples: {self.evaluation.max_samples}",
            f"  Batch Size: {self.evaluation.batch_size}",
            f"  Generation Samples: {self.evaluation.generation_num_samples}",
            f"  Detailed Metrics: {'✅' if self.evaluation.detailed_metrics else '❌'}"
        ]
        
        return "\n".join(lines)
    
    def save_to_file(self, filepath: Union[str, Path]):
        """Save current config as Python override file - UPDATED"""
        config_path = Path(filepath)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate Python config file
        content = [
            "# Generated configuration overrides",
            "# Edit this file to customize your experiment",
            "",
            "overrides = {",
        ]
        
        # Helper to add overrides for a section
        def add_section_overrides(section_name: str, section, defaults):
            for field in fields(section):
                current_value = getattr(section, field.name)
                default_value = getattr(defaults, field.name)
                
                if current_value != default_value:
                    if isinstance(current_value, str):
                        content.append(f'    "{section_name}.{field.name}": "{current_value}",')
                    else:
                        content.append(f'    "{section_name}.{field.name}": {current_value},')
        
        # Get defaults for comparison
        defaults = ProjectConfig.default()
        
        # Add top-level overrides
        for field in fields(self):
            if field.name in ['model', 'training', 'curriculum', 'generation', 'pipeline', 'evaluation']:
                continue  # Handle these specially
            
            current_value = getattr(self, field.name)
            default_value = getattr(defaults, field.name)
            
            if current_value != default_value:
                if isinstance(current_value, str):
                    content.append(f'    "{field.name}": "{current_value}",')
                else:
                    content.append(f'    "{field.name}": {current_value},')
        
        # Add section overrides
        add_section_overrides("model", self.model, defaults.model)
        add_section_overrides("training", self.training, defaults.training)
        add_section_overrides("generation", self.generation, defaults.generation)
        add_section_overrides("evaluation", self.evaluation, defaults.evaluation)  # NEW
        # Note: curriculum and pipeline are complex, skip for now
        
        content.append("}")
        
        with open(config_path, 'w') as f:
            f.write('\n'.join(content))
        
        print(f"Config saved to: {config_path}")


# Convenience functions
def get_default_config() -> ProjectConfig:
    """Get default project configuration"""
    return ProjectConfig.default()

def get_debug_config() -> ProjectConfig:
    """Get debug project configuration"""
    return ProjectConfig.debug()

def get_comprehensive_config() -> ProjectConfig:
    """Get comprehensive research configuration - NEW"""
    return ProjectConfig.comprehensive()

def get_memory_efficient_config() -> ProjectConfig:
    """Get memory-efficient configuration - NEW"""
    return ProjectConfig.memory_efficient()

def load_config_with_overrides(config_file: Optional[str] = None, 
                              args: Optional[argparse.Namespace] = None,
                              **kwargs) -> ProjectConfig:
    """Load config with cascading overrides
    
    Priority: defaults < config_file < args < kwargs
    """
    config = ProjectConfig.default()
    
    if config_file:
        config = config.override_from_file(config_file)
    
    if args:
        config = config.override_from_args(args)
    
    if kwargs:
        config = config.override(**kwargs)
    
    # Validate final config
    if not config.validate():
        raise ValueError("Final configuration is invalid")
    
    return config

def load_config_from_args(args: argparse.Namespace) -> ProjectConfig:
    """Standardized config loading from command line arguments"""
    config = ProjectConfig.default()
    
    # Load from config file if specified
    if hasattr(args, 'config') and args.config:
        config = config.override_from_file(args.config)
    
    # Override with command line arguments
    config = config.override_from_args(args)
    
    # Validate final config
    if not config.validate():
        raise ValueError("Configuration validation failed")
    
    return config