"""
Model Architecture Configuration - COMPLETE REVISED VERSION
Defines transformer architecture for tiny diffusion model with full configurability
All previously hardcoded parameters are now configurable
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class ModelConfig:
    """Complete model architecture configuration with all parameters"""
    
    # ================================
    # CORE ARCHITECTURE
    # ================================
    d_model: int = 768                    # Hidden dimension
    n_layers: int = 12                    # Number of transformer layers
    n_heads: int = 12                     # Number of attention heads
    d_ff: int = 2048                      # Feed-forward dimension
    
    # ================================
    # VOCABULARY & SEQUENCE
    # ================================
    vocab_size: int = 4191               # Vocabulary size (compressed from 50k GPT-2)
    max_seq_len: int = 512                # Maximum sequence length
    
    # ================================
    # REGULARIZATION
    # ================================
    dropout: float = 0.1                  # General dropout rate
    attention_dropout: float = 0.1        # Attention-specific dropout
    
    # ================================
    # ACTIVATION & NORMALIZATION
    # ================================
    activation: str = "swiglu"            # Activation function: swiglu, gelu, relu
    norm_type: str = "rmsnorm"            # Normalization: layernorm, rmsnorm
    norm_eps: float = 1e-6                # Normalization epsilon
    
    # ================================
    # POSITIONAL ENCODING
    # ================================
    pos_encoding: str = "rope"            # Positional encoding: rope, learned, sinusoidal
    rope_base: int = 10000                # NEW: RoPE frequency base (was hardcoded)
    
    # ================================
    # ATTENTION CONFIGURATION
    # ================================
    attention_scale: Optional[float] = None  # NEW: Override default 1/√d_head scaling
    qk_norm: bool = False                 # NEW: Query-Key normalization (experimental)
    
    # ================================
    # WEIGHT INITIALIZATION
    # ================================
    init_std: float = 0.02                # NEW: Standard deviation for weight init (was hardcoded)
    use_bias: bool = False                # Remove bias terms for efficiency
    
    # ================================
    # DIFFUSION-SPECIFIC PARAMETERS
    # ================================
    mask_token_id: int = 0                # Will be set during tokenizer creation
    min_masks_per_sample: int = 1         # NEW: Minimum masks per training sample
    max_masking_rate: float = 0.95        # NEW: Maximum allowed masking rate
    
    # ================================
    # LOSS FUNCTION CONFIGURATION
    # ================================
    loss_type: str = "cross_entropy"      # NEW: Loss function type
    label_smoothing: float = 0.0          # NEW: Label smoothing for training
    focal_loss_alpha: float = 1.0         # NEW: Focal loss alpha parameter
    focal_loss_gamma: float = 2.0         # NEW: Focal loss gamma parameter
    
    # ================================
    # MEMORY OPTIMIZATION
    # ================================
    gradient_checkpointing: bool = True   # Enable gradient checkpointing
    attention_cache_size: int = 2048      # NEW: RoPE cache size limit
    
    # ================================
    # EXPERIMENTAL FEATURES
    # ================================
    layerwise_lr_decay: bool = False      # NEW: Enable layer-wise learning rate decay
    layerwise_lr_decay_factor: float = 0.9  # NEW: Decay factor for layer-wise LR
    adaptive_attention_span: bool = False # NEW: Experimental adaptive attention
    attention_head_pruning: bool = False  # NEW: Enable attention head pruning
    
    @property
    def param_count(self) -> int:
        """Estimate parameter count using architectural formula"""
        # Core transformer parameters: P = 4lh² + 3lh·hf + 6lh + Vh
        attention_params = 4 * self.n_layers * self.d_model**2
        ffn_params = 3 * self.n_layers * self.d_model * self.d_ff  
        norm_params = 6 * self.n_layers * self.d_model
        embed_params = self.vocab_size * self.d_model
        
        total_params = attention_params + ffn_params + norm_params + embed_params
        
        # Add bias parameters if enabled
        if self.use_bias:
            bias_params = (
                4 * self.n_layers * self.d_model +    # Attention projections
                2 * self.n_layers * self.d_ff +       # FFN projections
                self.vocab_size                       # Output head bias
            )
            total_params += bias_params
        
        return total_params
    
    @property 
    def head_dim(self) -> int:
        """Calculate attention head dimension"""
        return self.d_model // self.n_heads
    
    @property
    def memory_footprint_mb(self) -> float:
        """Estimate memory footprint in MB (float32)"""
        return self.param_count * 4 / (1024 * 1024)
    
    @property
    def attention_complexity(self) -> str:
        """Get attention complexity description"""
        complexity = self.max_seq_len ** 2 * self.n_heads
        if complexity < 1e6:
            return f"{complexity/1e3:.1f}K operations"
        else:
            return f"{complexity/1e6:.1f}M operations"
    
    # ================================
    # PREDEFINED CONFIGURATIONS
    # ================================
    
    @classmethod
    def tiny_125m(cls) -> 'ModelConfig':
        """Configuration for ~125M parameter model"""
        return cls(
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_ff=2048,
            vocab_size=25000,
            max_seq_len=512,
            # All other parameters use defaults
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
            max_seq_len=512,
        )
    
    @classmethod 
    def medium_750m(cls) -> 'ModelConfig':
        """Configuration for ~750M parameter model"""
        return cls(
            d_model=1280,
            n_layers=20,
            n_heads=20,
            d_ff=3584,
            vocab_size=25000,
            max_seq_len=512,
        )
    
    @classmethod
    def experimental_125m(cls) -> 'ModelConfig':
        """Experimental configuration with advanced features"""
        return cls(
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_ff=2048,
            vocab_size=25000,
            max_seq_len=512,
            # Experimental features enabled
            qk_norm=True,
            attention_scale=0.125,  # Custom scaling
            layerwise_lr_decay=True,
            adaptive_attention_span=True,
            label_smoothing=0.1,
            loss_type="focal"
        )
    
    @classmethod
    def memory_efficient_125m(cls) -> 'ModelConfig':
        """Memory-efficient configuration for limited hardware"""
        return cls(
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_ff=1536,  # Reduced FFN size
            vocab_size=20000,  # Smaller vocabulary
            max_seq_len=256,   # Shorter sequences
            gradient_checkpointing=True,
            attention_cache_size=1024,  # Smaller cache
            dropout=0.15,  # Higher dropout for regularization
        )
    
    @classmethod
    def research_config(cls) -> 'ModelConfig':
        """Configuration optimized for research experiments"""
        return cls(
            d_model=512,   # Smaller for faster experiments
            n_layers=8,
            n_heads=8,
            d_ff=1024,
            vocab_size=15000,
            max_seq_len=256,
            # Research-friendly settings
            qk_norm=True,
            layerwise_lr_decay=True,
            attention_head_pruning=True,
            label_smoothing=0.05,
            init_std=0.015,  # Slightly smaller init
        )
    
    # ================================
    # VALIDATION & UTILITIES
    # ================================
    
    def validate(self) -> bool:
        """Comprehensive configuration validation"""
        checks = []
        
        # Basic architectural constraints
        checks.append(("d_model > 0", self.d_model > 0))
        checks.append(("n_layers > 0", self.n_layers > 0))
        checks.append(("n_heads > 0", self.n_heads > 0))
        checks.append(("d_ff > 0", self.d_ff > 0))
        checks.append(("d_model % n_heads == 0", self.d_model % self.n_heads == 0))
        
        # Vocabulary and sequence constraints
        checks.append(("vocab_size > 0", self.vocab_size > 0))
        checks.append(("max_seq_len > 0", self.max_seq_len > 0))
        checks.append(("vocab_size >= 1000", self.vocab_size >= 1000))  # Minimum viable vocab
        
        # Regularization constraints
        checks.append(("0 <= dropout <= 1", 0 <= self.dropout <= 1))
        checks.append(("0 <= attention_dropout <= 1", 0 <= self.attention_dropout <= 1))
        checks.append(("norm_eps > 0", self.norm_eps > 0))
        
        # Initialization constraints
        checks.append(("init_std > 0", self.init_std > 0))
        checks.append(("init_std < 1", self.init_std < 1))  # Reasonable range
        
        # RoPE constraints
        checks.append(("rope_base > 0", self.rope_base > 0))
        checks.append(("rope_base >= 1000", self.rope_base >= 1000))  # Reasonable minimum
        
        # Diffusion-specific constraints
        checks.append(("min_masks_per_sample >= 1", self.min_masks_per_sample >= 1))
        checks.append(("0 < max_masking_rate <= 1", 0 < self.max_masking_rate <= 1))
        
        # Loss function constraints
        checks.append(("0 <= label_smoothing < 1", 0 <= self.label_smoothing < 1))
        checks.append(("focal_loss_alpha > 0", self.focal_loss_alpha > 0))
        checks.append(("focal_loss_gamma >= 0", self.focal_loss_gamma >= 0))
        
        # String parameter validation
        valid_activations = ["swiglu", "gelu", "relu", "silu"]
        checks.append(("valid activation", self.activation in valid_activations))
        
        valid_norms = ["rmsnorm", "layernorm"]
        checks.append(("valid norm_type", self.norm_type in valid_norms))
        
        valid_pos_encodings = ["rope", "learned", "sinusoidal"]
        checks.append(("valid pos_encoding", self.pos_encoding in valid_pos_encodings))
        
        valid_loss_types = ["cross_entropy", "focal"]
        checks.append(("valid loss_type", self.loss_type in valid_loss_types))
        
        # Performance warnings
        if self.param_count > 1e9:
            print(f"⚠️  Warning: Model has {self.param_count/1e9:.1f}B parameters (may be too large)")
        
        # Memory warning
        if self.memory_footprint_mb > 2000:  # 2GB
            print(f"⚠️  Warning: Model requires {self.memory_footprint_mb:.0f}MB memory")
        
        # Embedding parameter ratio check
        embed_ratio = (self.vocab_size * self.d_model) / self.param_count
        if embed_ratio > 0.2:
            print(f"⚠️  Warning: Embedding parameters are {embed_ratio:.1%} of total "
                  f"(recommended <20%). Consider reducing vocab_size.")
        
        # Attention scale warning
        if self.attention_scale is not None:
            expected_scale = 1.0 / (self.head_dim ** 0.5)
            if abs(self.attention_scale - expected_scale) > 0.1:
                print(f"ℹ️  Info: Using custom attention scale {self.attention_scale:.4f} "
                      f"(default would be {expected_scale:.4f})")
        
        # Run all checks
        all_passed = True
        failed_checks = []
        
        for check_name, check_result in checks:
            if not check_result:
                failed_checks.append(check_name)
                all_passed = False
        
        if not all_passed:
            print(f"❌ Configuration validation failed: {failed_checks}")
        else:
            print("✅ Configuration validation passed")
        
        return all_passed
    
    def get_summary(self) -> str:
        """Get human-readable configuration summary"""
        lines = [
            f"Model Architecture Summary:",
            f"  Size: ~{self.param_count/1e6:.1f}M parameters ({self.memory_footprint_mb:.0f}MB)",
            f"  Architecture: {self.n_layers}L × {self.d_model}H × {self.n_heads}A",
            f"  Feed-forward: {self.d_ff} ({self.activation.upper()})",
            f"  Vocabulary: {self.vocab_size:,} tokens",
            f"  Sequence Length: {self.max_seq_len}",
            f"  Attention: {self.attention_complexity}",
            "",
            f"Configuration Details:",
            f"  Positional Encoding: {self.pos_encoding.upper()} (base={self.rope_base})",
            f"  Normalization: {self.norm_type.upper()} (eps={self.norm_eps})",
            f"  Initialization: σ={self.init_std}, bias={self.use_bias}",
            f"  Regularization: dropout={self.dropout}, attn_dropout={self.attention_dropout}",
            "",
            f"Diffusion Settings:",
            f"  Masking: {self.min_masks_per_sample}-{int(self.max_masking_rate*100)}% range",
            f"  Loss: {self.loss_type}" + (f" (smoothing={self.label_smoothing})" if self.label_smoothing > 0 else ""),
            "",
            f"Advanced Features:",
            f"  QK Normalization: {'✅' if self.qk_norm else '❌'}",
            f"  Custom Attention Scale: {'✅' if self.attention_scale else '❌'}",
            f"  Layer-wise LR Decay: {'✅' if self.layerwise_lr_decay else '❌'}",
            f"  Gradient Checkpointing: {'✅' if self.gradient_checkpointing else '❌'}",
        ]
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create from dictionary"""
        return cls(**config_dict)
    
    def copy(self, **overrides) -> 'ModelConfig':
        """Create a copy with optional parameter overrides"""
        config_dict = self.to_dict()
        config_dict.update(overrides)
        return self.from_dict(config_dict)
    
    def scale_model(self, scale_factor: float) -> 'ModelConfig':
        """Scale model size by a factor (useful for experiments)"""
        return self.copy(
            d_model=int(self.d_model * scale_factor),
            d_ff=int(self.d_ff * scale_factor),
            n_layers=max(1, int(self.n_layers * scale_factor)),
        )
    
    def get_comparable_configs(self) -> List[Tuple[str, 'ModelConfig']]:
        """Get list of comparable model configurations"""
        base_params = self.param_count
        
        configs = []
        
        # Generate variants with different trade-offs
        # Deeper model (more layers, smaller width)
        deeper = self.copy(
            n_layers=int(self.n_layers * 1.5),
            d_model=int(self.d_model * 0.8),
            d_ff=int(self.d_ff * 0.8)
        )
        configs.append(("deeper", deeper))
        
        # Wider model (fewer layers, larger width)  
        wider = self.copy(
            n_layers=max(1, int(self.n_layers * 0.7)),
            d_model=int(self.d_model * 1.2),
            d_ff=int(self.d_ff * 1.2)
        )
        configs.append(("wider", wider))
        
        # Experimental variant
        experimental = self.copy(
            qk_norm=True,
            layerwise_lr_decay=True,
            label_smoothing=0.1,
            attention_scale=1.0 / (self.head_dim ** 0.5) * 0.8
        )
        configs.append(("experimental", experimental))
        
        return configs