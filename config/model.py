"""
Model Architecture Configuration

Defines transformer architecture parameters and generation settings for the tiny 
masked diffusion language model. Follows 2025 research best practices:

- Deeper not wider for small models (12 layers, 768 dim for 125M params)
- Compressed tokenizer (<20% of parameters in embeddings)
- Bidirectional attention for masked diffusion
- RoPE positional embeddings and SwiGLU activation
"""

from typing import Dict, Any, Tuple, Optional
import math


def calculate_parameter_count(
    d_model: int,
    n_layers: int, 
    n_heads: int,
    vocab_size: int,
    sequence_length: int = 512
) -> Dict[str, int]:
    """
    Calculate parameter count for transformer architecture.
    
    Based on the formula from research:
    P = 4lh² + 3lh·hf + 6lh + Vh
    
    Where:
    - l = n_layers, h = d_model, hf = ffn_hidden_size
    - 4lh² = attention weights
    - 3lh·hf = SwiGLU MLP weights  
    - 6lh = layer norms
    - Vh = token embeddings
    """
    # Calculate SwiGLU FFN size (hardware-friendly)
    ffn_hidden_size = math.ceil(8 * d_model / 3 / 64) * 64
    
    attention_params = 4 * n_layers * d_model ** 2
    mlp_params = 3 * n_layers * d_model * ffn_hidden_size
    norm_params = 6 * n_layers * d_model  # RMSNorm, no bias
    embedding_params = vocab_size * d_model
    
    total = attention_params + mlp_params + norm_params + embedding_params
    
    return {
        'attention': attention_params,
        'mlp': mlp_params, 
        'norm': norm_params,
        'embedding': embedding_params,
        'total': total,
        'ffn_hidden_size': ffn_hidden_size
    }


def get_model_config() -> Dict[str, Any]:
    """
    Get default model configuration targeting ~125M parameters.
    
    Architecture follows research recommendations:
    - 12 layers (deeper not wider)
    - 768 hidden dim
    - Compressed vocab (~25k tokens)
    - Bidirectional attention for masked diffusion
    """
    config = {
        # Core architecture
        'd_model': 768,
        'n_layers': 12,
        'n_heads': 12,
        'vocab_size': 25000,  # Compressed from 50k GPT-2 vocab
        'max_position_embeddings': 2048,
        
        # Derived parameters
        'head_dim': None,  # Will be d_model // n_heads
        'ffn_hidden_size': None,  # Will be calculated
        
        # Activation and normalization
        'activation': 'swiglu',
        'norm_type': 'rmsnorm',
        'norm_eps': 1e-6,
        'use_bias': False,
        
        # Positional embeddings
        'position_embedding_type': 'rope',
        'rope_theta': 10000.0,
        'rope_scaling': None,
        
        # Attention
        'attention_dropout': 0.0,
        'hidden_dropout': 0.1,
        'use_causal_mask': False,  # Bidirectional for diffusion
        
        # Initialization
        'initializer_range': 0.02,
        'layer_norm_epsilon': 1e-6,
        
        # Training stability
        'gradient_checkpointing': True,
        'use_cache': False,  # Disable KV cache for training
        
        # Diffusion-specific
        'mask_token_id': None,  # Will be set by tokenizer
        'pad_token_id': None,   # Will be set by tokenizer
    }
    
    # Calculate derived parameters
    config['head_dim'] = config['d_model'] // config['n_heads']
    
    param_info = calculate_parameter_count(
        config['d_model'],
        config['n_layers'], 
        config['n_heads'],
        config['vocab_size']
    )
    config['ffn_hidden_size'] = param_info['ffn_hidden_size']
    config['parameter_count'] = param_info
    
    return config


def get_generation_config() -> Dict[str, Any]:
    """Generation parameters for inference and evaluation"""
    return {
        # Sampling parameters
        'temperature': 0.8,
        'top_p': 0.9,
        'top_k': 50,
        'repetition_penalty': 1.1,
        'length_penalty': 1.0,
        
        # Diffusion generation
        'num_diffusion_steps': 20,
        'guidance_scale': 1.0,  # For classifier-free guidance if used
        
        # Output control
        'max_new_tokens': 100,
        'min_new_tokens': 10,
        'do_sample': True,
        'num_return_sequences': 1,
        'return_dict_in_generate': True,
        
        # Special tokens
        'eos_token_id': None,  # Will be set by tokenizer
        'pad_token_id': None,  # Will be set by tokenizer
        'bos_token_id': None,  # Will be set by tokenizer
        
        # Early stopping
        'early_stopping': True,
        'use_cache': True,  # Enable for inference
    }


def validate_model_config(config: Dict[str, Any]) -> None:
    """Validate model configuration parameters"""
    assert config['d_model'] > 0, "d_model must be positive"
    assert config['n_layers'] > 0, "n_layers must be positive"  
    assert config['n_heads'] > 0, "n_heads must be positive"
    assert config['vocab_size'] > 0, "vocab_size must be positive"
    
    assert config['d_model'] % config['n_heads'] == 0, \
        f"d_model ({config['d_model']}) must be divisible by n_heads ({config['n_heads']})"
    
    # Check embedding parameter ratio
    param_info = config['parameter_count']
    embedding_ratio = param_info['embedding'] / param_info['total']
    assert embedding_ratio < 0.25, \
        f"Embedding parameters ({embedding_ratio:.1%}) exceed 25% of total. Use smaller vocab_size."
    
    print(f"Model validation passed:")
    print(f"  Total parameters: {param_info['total']:,}")
    print(f"  Embedding ratio: {embedding_ratio:.1%}")


def get_model_presets() -> Dict[str, Dict[str, Any]]:
    """Predefined model configurations for different use cases"""
    return {
        'tiny_125m': {
            'd_model': 768,
            'n_layers': 12, 
            'n_heads': 12,
            'vocab_size': 25000,
        },
        
        'small_350m': {
            'd_model': 1024,
            'n_layers': 16,
            'n_heads': 16, 
            'vocab_size': 32000,
        },
        
        'debug_7m': {
            'd_model': 128,
            'n_layers': 3,
            'n_heads': 4,
            'vocab_size': 5000,
        },
        
        'memory_efficient': {
            'd_model': 512,
            'n_layers': 10,
            'n_heads': 8,
            'vocab_size': 20000,
        }
    }


def get_model_config_by_preset(preset_name: str) -> Dict[str, Any]:
    """Get model configuration for a specific preset"""
    base_config = get_model_config()
    presets = get_model_presets()
    
    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    # Update base config with preset parameters
    preset_params = presets[preset_name]
    base_config.update(preset_params)
    
    # Recalculate derived parameters
    base_config['head_dim'] = base_config['d_model'] // base_config['n_heads']
    
    param_info = calculate_parameter_count(
        base_config['d_model'],
        base_config['n_layers'],
        base_config['n_heads'], 
        base_config['vocab_size']
    )
    base_config['ffn_hidden_size'] = param_info['ffn_hidden_size']
    base_config['parameter_count'] = param_info
    
    return base_config


def estimate_memory_requirements(config: Dict[str, Any], batch_size: int = 32, sequence_length: int = 512) -> Dict[str, float]:
    """
    Estimate memory requirements in GB for training and inference.
    
    Args:
        config: Model configuration
        batch_size: Training batch size
        sequence_length: Maximum sequence length
        
    Returns:
        Dictionary with memory estimates in GB
    """
    params = config['parameter_count']['total']
    d_model = config['d_model']
    n_layers = config['n_layers']
    
    # Model weights (fp16 for efficiency)
    model_memory_gb = params * 2 / 1e9
    
    # Activations (fp32 during training)
    # Rough estimate: batch_size * seq_len * d_model * n_layers * bytes_per_element
    activation_memory_gb = batch_size * sequence_length * d_model * n_layers * 4 / 1e9
    
    # Gradients (same size as model)
    gradient_memory_gb = model_memory_gb
    
    # Optimizer states (AdamW: 2x model params for momentum and variance)
    optimizer_memory_gb = model_memory_gb * 2
    
    # Total training memory
    training_total = model_memory_gb + activation_memory_gb + gradient_memory_gb + optimizer_memory_gb
    
    # Inference memory (model + activations only)
    inference_total = model_memory_gb + (activation_memory_gb * 0.3)  # Reduced activations
    
    return {
        'model_weights': model_memory_gb,
        'activations': activation_memory_gb,
        'gradients': gradient_memory_gb,
        'optimizer_states': optimizer_memory_gb,
        'training_total': training_total,
        'training_with_overhead': training_total * 1.3,  # 30% PyTorch overhead
        'inference_total': inference_total,
        'inference_with_overhead': inference_total * 1.2,  # 20% inference overhead
    }


def print_model_summary(config: Dict[str, Any]) -> None:
    """Print comprehensive model configuration summary"""
    param_info = config['parameter_count']
    
    print("=" * 60)
    print("MODEL CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print(f"Architecture:")
    print(f"  Model Dimension: {config['d_model']:,}")
    print(f"  Layers: {config['n_layers']}")
    print(f"  Attention Heads: {config['n_heads']}")
    print(f"  Head Dimension: {config['head_dim']}")
    print(f"  FFN Hidden Size: {config['ffn_hidden_size']:,}")
    print(f"  Vocabulary Size: {config['vocab_size']:,}")
    
    print(f"\nParameter Breakdown:")
    print(f"  Attention: {param_info['attention']:,} ({param_info['attention']/param_info['total']:.1%})")
    print(f"  MLP: {param_info['mlp']:,} ({param_info['mlp']/param_info['total']:.1%})")
    print(f"  LayerNorm: {param_info['norm']:,} ({param_info['norm']/param_info['total']:.1%})")
    print(f"  Embeddings: {param_info['embedding']:,} ({param_info['embedding']/param_info['total']:.1%})")
    print(f"  Total: {param_info['total']:,}")
    
    # Memory estimates for different batch sizes
    print(f"\nMemory Estimates (Training):")
    for batch_size in [8, 16, 32]:
        mem = estimate_memory_requirements(config, batch_size)
        print(f"  Batch {batch_size}: {mem['training_with_overhead']:.1f}GB")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test and demonstrate the configuration
    print("Testing model configurations...")
    
    # Test default config
    config = get_model_config()
    validate_model_config(config)
    print_model_summary(config)
    
    # Test presets
    print("\nTesting presets...")
    for preset_name in ['debug_7m', 'memory_efficient']:
        print(f"\n{preset_name.upper()}:")
        preset_config = get_model_config_by_preset(preset_name)
        validate_model_config(preset_config)
        
        params = preset_config['parameter_count']['total']
        mem_8gb = estimate_memory_requirements(preset_config, batch_size=16)
        fits_8gb = mem_8gb['training_with_overhead'] < 8.0
        
        print(f"  Parameters: {params:,}")
        print(f"  Memory (batch 16): {mem_8gb['training_with_overhead']:.1f}GB")
        print(f"  Fits 8GB GPU: {'✓' if fits_8gb else '✗'}")