"""
Transformer blocks with SwiGLU and RMSNorm
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
    """Single transformer block"""
    
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 use_bias: bool = False,
                 norm_eps: float = 1e-6):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=attention_dropout,
            use_bias=use_bias
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
    """Complete transformer model"""
    
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
                 pad_token_id: int = 0):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                attention_dropout=attention_dropout,
                use_bias=use_bias,
                norm_eps=norm_eps
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = RMSNorm(d_model, norm_eps)
        
        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=use_bias)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
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