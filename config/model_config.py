"""
Model Architecture Configuration
Defines transformer architecture for tiny diffusion model
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    
    # Core architecture
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 2048
    
    # Vocabulary & sequence  
    vocab_size: int = 25000  # Compressed from 50k GPT-2 vocab
    max_seq_len: int = 512
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Activation & norms
    activation: str = "swiglu"  # swiglu, gelu, relu
    norm_type: str = "rmsnorm"  # layernorm, rmsnorm
    norm_eps: float = 1e-6
    
    # Positional encoding
    pos_encoding: str = "rope"  # rope, learned, sinusoidal
    
    # Initialization
    init_std: float = 0.02
    use_bias: bool = False  # Remove bias terms for efficiency
    
    # Diffusion specific
    mask_token_id: int = 0  # Will be set during tokenizer creation
    
    @property
    def param_count(self) -> int:
        """Estimate parameter count"""
        # P = 4lh² + 3lh·hf + 6lh + Vh (from architecture formula)
        attention_params = 4 * self.n_layers * self.d_model**2
        ffn_params = 3 * self.n_layers * self.d_model * self.d_ff  
        norm_params = 6 * self.n_layers * self.d_model
        embed_params = self.vocab_size * self.d_model
        
        return attention_params + ffn_params + norm_params + embed_params
    
    @classmethod
    def tiny_125m(cls) -> 'ModelConfig':
        """Configuration for ~125M parameter model"""
        return cls(
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_ff=2048,
            vocab_size=25000,
            max_seq_len=512
        )
    
    @classmethod
    def small_350m(cls) -> 'ModelConfig':
        """Configuration for ~350M parameter model"""
        return cls(
            d_model=1024,
            n_layers=16,
            n_heads=16,
            d_ff=2816,
            vocab_size=25000,
            max_seq_len=512
        )
    
    def validate(self):
        """Validate configuration"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        
        # Check embedding parameter ratio (should be <20% of total)
        embed_ratio = (self.vocab_size * self.d_model) / self.param_count
        if embed_ratio > 0.2:
            print(f"Warning: Embedding parameters are {embed_ratio:.1%} of total "
                  f"(recommended <20%). Consider reducing vocab_size.")
        
        return True