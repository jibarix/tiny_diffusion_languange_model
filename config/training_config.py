"""
Training Configuration
Hyperparameters and training settings
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
            use_mixed_precision=True
        )
    
    @classmethod
    def memory_efficient(cls) -> 'TrainingConfig':
        """Memory-efficient config for limited VRAM"""
        return cls(
            batch_size=16,
            gradient_accumulation_steps=4,
            use_gradient_checkpointing=True,
            use_mixed_precision=True,
            dataloader_num_workers=2
        )
    
    @classmethod
    def fast_debug(cls) -> 'TrainingConfig':
        """Fast config for debugging"""
        return cls(
            batch_size=8,
            max_epochs=2,
            eval_every=50,
            save_every=100,
            log_every=10
        )
    
    def validate(self, model_config=None):
        """Validate training configuration"""
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert 0 <= self.val_split < 1, "val_split must be in [0, 1)"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        
        # Memory usage estimation (if model config provided)
        if model_config is not None:
            model_params_mb = model_config.param_count / 1e6
            batch_mem_mb = self.batch_size * model_config.max_seq_len * 4 / 1e6  # 4 bytes per token
            estimated_vram_gb = (model_params_mb * 3 + batch_mem_mb) / 1000  # 3x for model+optim+grads
            
            if estimated_vram_gb > 8:
                print(f"Warning: Estimated VRAM usage {estimated_vram_gb:.1f}GB may exceed 8GB limit")
        
        return True