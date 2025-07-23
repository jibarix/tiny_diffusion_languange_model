"""
Transformer blocks with SwiGLU and RMSNorm - REVISED WITH CONFIG SUPPORT
All hardcoded parameters moved to configuration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return self.weight * x_normed


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, d_model: int, d_ff: int, use_bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=use_bias)  # Gate
        self.w2 = nn.Linear(d_model, d_ff, bias=use_bias)  # Up
        self.w3 = nn.Linear(d_ff, d_model, bias=use_bias)  # Down
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    """Single transformer block with configurable parameters"""
    
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 use_bias: bool = False,
                 norm_eps: float = 1e-6,
                 max_seq_len: int = 2048,
                 rope_base: int = 10000):  # NEW: Accept rope_base parameter
        super().__init__()
        
        # Multi-head attention with configurable RoPE base
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=attention_dropout,
            use_bias=use_bias,
            max_seq_len=max_seq_len,
            rope_base=rope_base  # NEW: Pass configurable rope_base
        )
        
        # Feed-forward network
        self.ffn = SwiGLU(d_model, d_ff, use_bias)
        
        # Layer norms
        self.norm1 = RMSNorm(d_model, norm_eps)
        self.norm2 = RMSNorm(d_model, norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor,
                attn_mask=None,
                key_padding_mask=None) -> torch.Tensor:
        """Forward pass with residual connections"""
        
        # Self-attention + residual
        normed_x = self.norm1(x)
        attn_out = self.attention(normed_x, attn_mask, key_padding_mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward + residual  
        normed_x = self.norm2(x)
        ffn_out = self.ffn(normed_x)
        x = x + self.dropout(ffn_out)
        
        return x


class TransformerModel(nn.Module):
    """Complete transformer model with full configuration support"""
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 d_ff: int,
                 max_seq_len: int = 2048,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 use_bias: bool = False,
                 norm_eps: float = 1e-6,
                 pad_token_id: int = 0,
                 init_std: float = 0.02,      # NEW: Configurable weight initialization
                 rope_base: int = 10000):     # NEW: Configurable RoPE frequency base
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
        # Store configuration parameters
        self.init_std = init_std
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Transformer layers with configurable parameters
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                attention_dropout=attention_dropout,
                use_bias=use_bias,
                norm_eps=norm_eps,
                max_seq_len=max_seq_len,
                rope_base=rope_base  # NEW: Pass configurable rope_base
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = RMSNorm(d_model, norm_eps)
        
        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=use_bias)
        
        # Initialize weights using configurable parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using configurable parameters"""
        if isinstance(module, nn.Linear):
            # FIXED: Use configurable init_std instead of hardcoded 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # FIXED: Use configurable init_std instead of hardcoded 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            # Keep padding token embedding at zero
            if hasattr(self, 'pad_token_id') and self.pad_token_id is not None:
                with torch.no_grad():
                    module.weight[self.pad_token_id].fill_(0)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                causal_mask: bool = False) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] (True = keep, False = mask)
            causal_mask: Whether to apply causal attention (for AR training)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        
        # Create attention masks
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()  # Invert: True = mask
            
        attn_mask = None
        if causal_mask:
            # Lower triangular mask for causal attention
            attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask)
        
        # Final norm
        x = self.final_norm(x)
        
        # Output projection
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def get_num_params(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_config_dict(self) -> dict:
        """Get configuration dictionary for model saving/loading"""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_layers': len(self.layers),
            'max_seq_len': self.max_seq_len,
            'init_std': self.init_std,
            'rope_base': self.rope_base,
            'pad_token_id': self.pad_token_id
        }
    
    def resize_token_embeddings(self, new_vocab_size: int):
        """Resize token embeddings to new vocabulary size"""
        if new_vocab_size == self.vocab_size:
            return
        
        old_embeddings = self.token_embedding.weight.data
        old_lm_head = self.lm_head.weight.data
        
        # Create new embedding layer
        new_token_embedding = nn.Embedding(
            new_vocab_size, self.d_model, 
            padding_idx=self.pad_token_id
        )
        
        # Create new output head
        new_lm_head = nn.Linear(self.d_model, new_vocab_size, bias=False)
        
        # Copy old weights
        min_vocab_size = min(self.vocab_size, new_vocab_size)
        new_token_embedding.weight.data[:min_vocab_size] = old_embeddings[:min_vocab_size]
        new_lm_head.weight.data[:min_vocab_size] = old_lm_head[:min_vocab_size]
        
        # Initialize new tokens if vocabulary expanded
        if new_vocab_size > self.vocab_size:
            with torch.no_grad():
                # Initialize new embedding tokens
                torch.nn.init.normal_(
                    new_token_embedding.weight.data[self.vocab_size:], 
                    mean=0.0, std=self.init_std
                )
                # Initialize new output head tokens
                torch.nn.init.normal_(
                    new_lm_head.weight.data[self.vocab_size:], 
                    mean=0.0, std=self.init_std
                )
        
        # Replace layers
        self.token_embedding = new_token_embedding
        self.lm_head = new_lm_head
        self.vocab_size = new_vocab_size
        
        # Move to same device as original model
        device = next(self.parameters()).device
        self.token_embedding.to(device)
        self.lm_head.to(device)
        
        print(f"Resized token embeddings: {old_embeddings.size(0)} -> {new_vocab_size}")
    
    def get_memory_footprint(self) -> dict:
        """Get estimated memory footprint breakdown"""
        def get_param_size(module):
            return sum(p.numel() * 4 for p in module.parameters()) / 1024**2  # MB
        
        return {
            'token_embedding_mb': get_param_size(self.token_embedding),
            'transformer_layers_mb': sum(get_param_size(layer) for layer in self.layers),
            'output_head_mb': get_param_size(self.lm_head),
            'total_mb': get_param_size(self),
            'total_params': self.get_num_params()
        }
    
    def freeze_embeddings(self):
        """Freeze token embeddings for fine-tuning"""
        for param in self.token_embedding.parameters():
            param.requires_grad = False
        print("Token embeddings frozen")
    
    def unfreeze_embeddings(self):
        """Unfreeze token embeddings"""
        for param in self.token_embedding.parameters():
            param.requires_grad = True
        print("Token embeddings unfrozen")
    
    def freeze_layers(self, layer_indices: list):
        """Freeze specific transformer layers"""
        for idx in layer_indices:
            if 0 <= idx < len(self.layers):
                for param in self.layers[idx].parameters():
                    param.requires_grad = False
        print(f"Frozen layers: {layer_indices}")
    
    def unfreeze_all_layers(self):
        """Unfreeze all transformer layers"""
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = True
        print("All layers unfrozen")
    
    def get_layer_wise_lr_groups(self, base_lr: float, decay_factor: float = 0.9):
        """Get parameter groups for layer-wise learning rate decay"""
        param_groups = []
        
        # Embedding layer (highest LR)
        param_groups.append({
            'params': list(self.token_embedding.parameters()),
            'lr': base_lr,
            'layer_name': 'embeddings'
        })
        
        # Transformer layers (decreasing LR)
        for i, layer in enumerate(self.layers):
            layer_lr = base_lr * (decay_factor ** (len(self.layers) - i - 1))
            param_groups.append({
                'params': list(layer.parameters()),
                'lr': layer_lr,
                'layer_name': f'layer_{i}'
            })
        
        # Output head (lowest LR)
        param_groups.append({
            'params': list(self.lm_head.parameters()) + list(self.final_norm.parameters()),
            'lr': base_lr * (decay_factor ** len(self.layers)),
            'layer_name': 'output_head'
        })
        
        return param_groups
    
    def get_attention_weights(self, input_ids: torch.Tensor, layer_idx: int = -1):
        """Extract attention weights from specified layer for analysis"""
        if not (-len(self.layers) <= layer_idx < len(self.layers)):
            raise ValueError(f"Layer index {layer_idx} out of range")
        
        # Forward pass up to specified layer
        x = self.token_embedding(input_ids)
        
        target_layer_idx = layer_idx if layer_idx >= 0 else len(self.layers) + layer_idx
        
        for i, layer in enumerate(self.layers):
            if i == target_layer_idx:
                # Get attention weights from this layer
                normed_x = layer.norm1(x)
                # Note: This would require modifying MultiHeadAttention to return weights
                # For now, just return None as placeholder
                return None
            x = layer(x)
        
        return None