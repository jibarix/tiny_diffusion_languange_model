"""
Unified Configuration System for Tiny Text Diffusion Model

Provides a single dataclass-based configuration with override cascading:
command line → config file → programmatic overrides

Usage:
    from config import ProjectConfig
    
    # Default configuration
    config = ProjectConfig.default()
    
    # Override with dot notation
    config = config.override(**{
        "model.d_model": 512,
        "training.batch_size": 16,
        "curriculum.stages[0].epochs": 25
    })
    
    # Load from config file
    config = config.override_from_file("experiment.py")
    
    # Quick presets
    debug_config = ProjectConfig.debug()
    memory_config = ProjectConfig.memory_efficient()
"""

import argparse
import copy
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
import json


def nested_set(dictionary: Dict, key_path: str, value: Any) -> None:
    """Set nested dictionary value using dot notation (e.g., 'model.d_model')"""
    keys = key_path.split('.')
    current = dictionary
    
    for key in keys[:-1]:
        # Handle list indexing like 'stages[0]'
        if '[' in key and ']' in key:
            base_key, index_str = key.split('[', 1)
            index = int(index_str.rstrip(']'))
            if base_key not in current:
                current[base_key] = []
            while len(current[base_key]) <= index:
                current[base_key].append({})
            current = current[base_key][index]
        else:
            if key not in current:
                current[key] = {}
            current = current[key]
    
    final_key = keys[-1]
    if '[' in final_key and ']' in final_key:
        base_key, index_str = final_key.split('[', 1)
        index = int(index_str.rstrip(']'))
        if base_key not in current:
            current[base_key] = []
        while len(current[base_key]) <= index:
            current[base_key].append(None)
        current[base_key][index] = value
    else:
        current[final_key] = value


def nested_get(dictionary: Dict, key_path: str, default: Any = None) -> Any:
    """Get nested dictionary value using dot notation"""
    keys = key_path.split('.')
    current = dictionary
    
    try:
        for key in keys:
            if '[' in key and ']' in key:
                base_key, index_str = key.split('[', 1)
                index = int(index_str.rstrip(']'))
                current = current[base_key][index]
            else:
                current = current[key]
        return current
    except (KeyError, IndexError, TypeError):
        return default


@dataclass
class ProjectConfig:
    """
    Unified configuration for the tiny text diffusion model project.
    
    This serves as the single source of truth for all configuration parameters,
    with support for hierarchical overrides and preset configurations.
    """
    
    # Model configuration (imported from config.model)
    model: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    training: Dict[str, Any] = field(default_factory=dict)
    
    # Curriculum configuration (imported from config.curriculum)
    curriculum: Dict[str, Any] = field(default_factory=dict)
    
    # Data configuration
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation configuration
    evaluation: Dict[str, Any] = field(default_factory=dict)
    
    # System configuration
    system: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def default(cls) -> 'ProjectConfig':
        """Create default configuration with research-optimized parameters"""
        from .model import get_model_config
        from .curriculum import get_curriculum_config
        
        # Get curriculum config to calculate total epochs
        curriculum_config = get_curriculum_config()
        total_epochs = sum(stage['epochs'] for stage in curriculum_config['stages'])
        
        # --- MODIFICATION FOR CONTINUED TRAINING ---
        # Get the base model config and apply the fixed masking rate.
        model_config = get_model_config()
        model_config['fixed_masking_rate'] = 0.15 # Set to 15% for stable fine-tuning
        
        return cls(
            model=model_config,
            training={
                # Core optimization parameters (2025 research-validated)
                'batch_size': 32,
                'learning_rate': 1e-5,
                'weight_decay': 0.01,
                'warmup_steps': 1500,
                'gradient_clipping': 1.0,
                'optimizer': 'AdamW',
                'scheduler': 'cosine_with_restarts',
                
                # Advanced optimization features
                'label_smoothing': 0.1,
                'gradient_accumulation_steps': 2,
                'mixed_precision': True,
                'gradient_checkpointing': True,
                
                # Training control
                'max_epochs': total_epochs, # Calculated from curriculum epochs
                'save_every': 10,
                'eval_every': 5,
                'early_stopping_patience': 30, # MODIFIED: Increased for longer convergence
            },
            curriculum=curriculum_config,
            data={
                'sequence_length': 512,
                'min_sentence_length': 10,
                'max_sentence_length': 200,
                'vocab_compression_target': 0.9,  # Cover 90% of corpus
                'validation_split': 0.1,
                'random_seed': 42,
            },
            evaluation={
                # Optimized generation parameters
                'generation_length': 100,
                'temperature': 0.6,
                'top_p': 0.85,
                'top_k': 20,
                'num_samples': 5,
                'perplexity_eval_size': 1000,
            },
            system={
                'device': 'auto',  # auto-detect GPU/CPU
                'num_workers': 0,
                'pin_memory': False,
                'compile_model': False,  # PyTorch 2.0 compilation
                'memory_efficient': True,  # Enable memory optimizations
            }
        )
    
    @classmethod
    def debug(cls) -> 'ProjectConfig':
        """Create debug configuration for fast testing"""
        config = cls.default()
        
        # Use full model (remove tiny model overrides)
        # config.model remains default: 768d/12L/12H/25k vocab
        
        # Minimal curriculum
        config.curriculum['stages'][0]['epochs'] = 1
        config.curriculum['stages'][1]['epochs'] = 2
        config.curriculum['stages'][2]['epochs'] = 3
        
        # Calculate total epochs from curriculum
        total_epochs = sum(stage['epochs'] for stage in config.curriculum['stages'])
        
        # Fast training with optimized parameters
        config.training.update({
            'batch_size': 32,
            'learning_rate': 1e-5,
            'label_smoothing': 0.1,
            'gradient_accumulation_steps': 2,
            'max_epochs': total_epochs,  # Calculated
            'save_every': 1,
            'eval_every': 1,
        })
        
        # Small data for speed
        config.data.update({
            'sequence_length': 512,
            'validation_split': 0.1,
        })
        
        # Keep optimized generation parameters
        config.evaluation.update({
            'temperature': 0.6,
            'top_k': 20,
            'top_p': 0.85,
        })
        
        return config
    
    @classmethod
    def memory_efficient(cls) -> 'ProjectConfig':
        """Create memory-efficient configuration for 8GB VRAM"""
        config = cls.default()
        
        # Memory optimizations (already applied in default)
        config.training.update({
            'batch_size': 4,
            'gradient_accumulation_steps': 4,
            'gradient_checkpointing': True,
            'mixed_precision': True,
        })
        
        # Smaller model variant
        config.model.update({
            'd_model': 512,
            'n_layers': 10,
        })
        
        # Recalculate head_dim
        config.model['head_dim'] = config.model['d_model'] // config.model['n_heads']
        
        config.system.update({
            'memory_efficient': True,
            'num_workers': 2,
        })
        
        return config
    
    @classmethod
    def test_integration(cls) -> 'ProjectConfig':
        """Create configuration for integration testing (30s runtime)"""
        config = cls.debug()
        
        # Slightly more realistic than debug but still fast
        config.training.update({
            'batch_size': 8,
            'learning_rate': 1e-4,
            'label_smoothing': 0.1,
            'max_epochs': 6,
        })
        
        config.curriculum['stages'][0]['epochs'] = 2
        config.curriculum['stages'][1]['epochs'] = 2
        config.curriculum['stages'][2]['epochs'] = 2
        
        config.data['sequence_length'] = 256
        
        return config
    
    def override(self, **kwargs) -> 'ProjectConfig':
        """
        Create new config with overrides using dot notation.
        
        Args:
            **kwargs: Overrides in dot notation, e.g.:
                model.d_model=512, 
                training.batch_size=16,
                curriculum.stages[0].epochs=25
        
        Returns:
            New ProjectConfig instance with overrides applied
        """
        # Deep copy to avoid modifying original
        new_config = copy.deepcopy(self)
        config_dict = asdict(new_config)
        
        # Apply overrides
        for key_path, value in kwargs.items():
            nested_set(config_dict, key_path, value)
        
        # Reconstruct config object
        return ProjectConfig(**config_dict)
    
    def override_from_file(self, config_file: str) -> 'ProjectConfig':
        """
        Load overrides from a Python file.
        
        The file should define overrides as a dictionary, e.g.:
        ```python
        # experiment.py
        overrides = {
            "model.d_model": 512,
            "training.batch_size": 16,
            "curriculum.stages[0].epochs": 25
        }
        ```
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # Execute the config file to get overrides
        namespace = {}
        exec(open(config_file).read(), namespace)
        
        if 'overrides' not in namespace:
            raise ValueError(f"Config file {config_file} must define 'overrides' dictionary")
        
        return self.override(**namespace['overrides'])
    
    def override_from_args(self, args: Optional[argparse.Namespace] = None) -> 'ProjectConfig':
        """Override configuration from command line arguments"""
        if args is None:
            return self
        
        overrides = {}
        
        # Map common CLI arguments to config paths
        arg_mappings = {
            'batch_size': 'training.batch_size',
            'learning_rate': 'training.learning_rate',
            'epochs': 'training.max_epochs',
            'd_model': 'model.d_model',
            'n_layers': 'model.n_layers',
            'sequence_length': 'data.sequence_length',
            'memory_efficient': 'system.memory_efficient',
        }
        
        for arg_name, config_path in arg_mappings.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                overrides[config_path] = getattr(args, arg_name)
        
        # Handle any additional overrides passed as --override key=value
        if hasattr(args, 'override') and args.override:
            for override_str in args.override:
                key, value = override_str.split('=', 1)
                # Try to parse as JSON, fall back to string
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass  # Keep as string
                overrides[key] = value
        
        return self.override(**overrides) if overrides else self
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        # Model validation
        assert self.model['d_model'] > 0, "d_model must be positive"
        assert self.model['n_layers'] > 0, "n_layers must be positive"
        assert self.model['n_heads'] > 0, "n_heads must be positive"
        assert self.model['d_model'] % self.model['n_heads'] == 0, \
            "d_model must be divisible by n_heads"
        
        # Training validation
        assert 0 < self.training['learning_rate'] < 1, "Invalid learning rate"
        assert self.training['batch_size'] > 0, "batch_size must be positive"
        assert 0 <= self.training['weight_decay'] <= 1, "weight_decay must be in [0,1]"
        assert 0 <= self.training.get('label_smoothing', 0.1) <= 1, "label_smoothing must be in [0,1]"
        
        # Curriculum validation
        total_epochs = sum(stage['epochs'] for stage in self.curriculum['stages'])
        self.training['max_epochs'] = total_epochs # Sync max_epochs with curriculum
        
        # Data validation
        assert 0 < self.data['validation_split'] < 1, "validation_split must be in (0,1)"
        assert self.data['sequence_length'] > 0, "sequence_length must be positive"
        
        # Generation validation
        assert 0 < self.evaluation['temperature'] <= 2.0, "temperature must be in (0,2]"
        assert 0 < self.evaluation['top_p'] <= 1.0, "top_p must be in (0,1]"
        assert self.evaluation['top_k'] > 0, "top_k must be positive"
    
    def estimate_memory_usage(self) -> Dict[str, float]:
        """Estimate GPU memory usage in GB"""
        # Approximate memory calculation for transformer
        params = self.model['d_model'] ** 2 * self.model['n_layers'] * 12  # Rough estimate
        
        # Model weights (fp16)
        model_memory = params * 2 / 1e9  # 2 bytes per param in fp16
        
        # Activations (depends on batch size and sequence length)
        effective_batch_size = (
            self.training['batch_size'] * self.training.get('gradient_accumulation_steps', 1)
        )
        activation_memory = (
            effective_batch_size * self.data['sequence_length'] * self.model['d_model'] * self.model['n_layers'] * 4 / 1e9  # 4 bytes per activation in fp32
        )
        
        # Gradients (same size as model)
        gradient_memory = model_memory
        
        # Optimizer states (AdamW stores 2x model params)
        optimizer_memory = model_memory * 2
        
        total = model_memory + activation_memory + gradient_memory + optimizer_memory
        
        return {
            'model': model_memory,
            'activations': activation_memory,
            'gradients': gradient_memory,
            'optimizer': optimizer_memory,
            'total': total,
            'total_with_overhead': total * 1.2,  # 20% overhead for PyTorch
            'effective_batch_size': effective_batch_size,
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        return nested_get(asdict(self), key_path, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ProjectConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser for common overrides"""
    parser = argparse.ArgumentParser(description='Tiny Text Diffusion Model')
    
    # Preset configurations
    presets = parser.add_mutually_exclusive_group()
    presets.add_argument('--debug', action='store_true', help='Use debug configuration')
    presets.add_argument('--test', action='store_true', help='Use test configuration (10s)')
    presets.add_argument('--test-integration', action='store_true', help='Use integration test (30s)')
    presets.add_argument('--memory-efficient', action='store_true', help='Use memory-efficient configuration')
    
    # Common overrides
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Maximum training epochs')
    parser.add_argument('--d-model', type=int, help='Model dimension')
    parser.add_argument('--n-layers', type=int, help='Number of transformer layers')
    parser.add_argument('--sequence-length', type=int, help='Maximum sequence length')
    
    # Generic override mechanism
    parser.add_argument('--override', action='append', help='Override config value: key=value')
    
    # Config file
    parser.add_argument('--config', type=str, help='Load overrides from Python file')
    
    return parser


def get_config_from_args(args: Optional[argparse.Namespace] = None) -> ProjectConfig:
    """
    Get configuration from command line arguments.
    
    Args:
        args: Parsed arguments. If None, will parse sys.argv
    
    Returns:
        ProjectConfig instance
    """
    if args is None:
        parser = create_argument_parser()
        args = parser.parse_args()
    
    # Start with appropriate preset
    if args.debug:
        config = ProjectConfig.debug()
    elif args.test:
        config = ProjectConfig.debug()  # test is same as debug for now
    elif args.test_integration:
        config = ProjectConfig.test_integration()
    elif args.memory_efficient:
        config = ProjectConfig.memory_efficient()
    else:
        config = ProjectConfig.default()
    
    # Apply config file overrides
    if hasattr(args, 'config') and args.config:
        config = config.override_from_file(args.config)
    
    # Apply command line overrides
    config = config.override_from_args(args)
    
    # Validate final configuration
    config.validate()
    
    return config
