"""
Training Configuration - FIXED VERSION
Removes hardcoded values and makes them configurable
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings"""
    
    # Optimization
    learning_rate: float = 2e-4
    min_learning_rate: float = 2e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip_norm: float = 1.0
    
    # Learning rate schedule
    warmup_steps: int = 1000
    scheduler: str = "cosine_with_restarts"  # cosine, linear, constant
    
    # Batch settings
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Training length
    max_epochs: int = 300
    max_steps: Optional[int] = None
    
    # Evaluation & checkpointing
    eval_every: int = 1000
    save_every: int = 5000
    log_every: int = 100
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Validation
    val_split: float = 0.1
    val_batch_size: Optional[int] = None  # Use batch_size if None
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    
    # Regularization
    label_smoothing: float = 0.0
    dropout_schedule: str = "constant"  # constant, decay
    
    # NEW: Hardware configuration (previously hardcoded)
    max_vram_gb: float = 8.0              # VRAM limit (was hardcoded as 8)
    memory_safety_margin: float = 0.1     # Safety margin for memory estimation
    bytes_per_param: int = 4              # Bytes per parameter (was hardcoded as 4)
    memory_multiplier: float = 3.0        # Memory multiplier for model+optimizer+gradients (was hardcoded as 3)

    # NEW: Dynamic difficulty parameters (from trainer.py)
    loss_normalization_factor: float = 10.0  # "max reasonable loss ~10"
    gradient_normalization_factor: float = 1.0  # "max grad ~1.0"
    
    # NEW: Validation parameters (from trainer.py)
    validation_masking_rate: float = 0.15    # Fixed rate for validation
    validation_frequency: int = 5            # Validate every N epochs
    
    # NEW: Curriculum scheduler parameters (from trainer.py)
    adaptation_window: int = 10
    min_epochs_per_stage: int = 5
    max_epochs_per_stage: int = 200
    performance_plateau_epochs: int = 10
    improvement_threshold: float = 0.01
    
    @property
    def effective_batch_size(self) -> int:
        """Effective batch size including gradient accumulation"""
        return self.batch_size * self.gradient_accumulation_steps
    
    @property
    def val_batch_size_actual(self) -> int:
        """Actual validation batch size"""
        return self.val_batch_size or self.batch_size
    
    @classmethod
    def default(cls) -> 'TrainingConfig':
        """Default configuration for 8GB VRAM"""
        return cls(
            batch_size=32,
            gradient_accumulation_steps=2,
            use_gradient_checkpointing=True,
            use_mixed_precision=True,
            max_vram_gb=8.0,  # Explicitly set instead of hardcoded
        )
    
    @classmethod
    def memory_efficient(cls) -> 'TrainingConfig':
        """Memory-efficient config for limited VRAM"""
        return cls(
            batch_size=16,
            gradient_accumulation_steps=4,
            use_gradient_checkpointing=True,
            use_mixed_precision=True,
            dataloader_num_workers=2,
            max_vram_gb=8.0,
            pin_memory=False
        )
    
    @classmethod
    def fast_debug(cls) -> 'TrainingConfig':
        """Fast config for debugging"""
        return cls(
            batch_size=8,
            max_epochs=2,
            eval_every=50,
            save_every=100,
            log_every=10,
            max_vram_gb=8.0,  # Configure instead of hardcode
        )
    
    def validate(self, model_config=None):
        """Validate training configuration with configurable hardware limits"""
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert 0 <= self.val_split < 1, "val_split must be in [0, 1)"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        
        # Validate hardware configuration
        assert self.max_vram_gb > 0, "max_vram_gb must be positive"
        assert 0 <= self.memory_safety_margin <= 1, "memory_safety_margin must be between 0 and 1"
        assert self.bytes_per_param > 0, "bytes_per_param must be positive"
        assert self.memory_multiplier > 0, "memory_multiplier must be positive"
        
        # Memory usage estimation with CONFIGURABLE values
        if model_config is not None:
            # Use configurable values instead of hardcoded ones
            model_params_mb = model_config.param_count * self.bytes_per_param / 1e6
            batch_mem_mb = (self.batch_size * model_config.max_seq_len * 
                           self.bytes_per_param / 1e6)
            
            # Apply configurable memory multiplier and convert to GB
            estimated_vram_gb = (model_params_mb * self.memory_multiplier + batch_mem_mb) / 1000
            
            # Apply safety margin to VRAM limit
            safe_vram_limit = self.max_vram_gb * (1 - self.memory_safety_margin)
            
            if estimated_vram_gb > safe_vram_limit:
                print(f"Warning: Estimated VRAM usage {estimated_vram_gb:.1f}GB may exceed "
                      f"safe limit {safe_vram_limit:.1f}GB (max: {self.max_vram_gb:.1f}GB)")
                print(f"Consider reducing batch_size, using gradient accumulation, or enabling "
                      f"gradient checkpointing and mixed precision")
        
        return True