"""
Multi-Head Attention with RoPE
Bidirectional attention for diffusion (no causal masking)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RoPEEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for efficiency
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0
    
    def _get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached or compute cos/sin for given sequence length"""
        if seq_len > self._cached_seq_len or self._cached_cos is None or self._cached_cos.device != device:
            # Compute new cache
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cached_cos = emb.cos()
            self._cached_sin = emb.sin()
            self._cached_seq_len = seq_len
        
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors"""
        seq_len = q.size(-2)
        cos, sin = self._get_cos_sin(seq_len, q.device)
        
        return self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)
    
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to tensor"""
        # Split into even/odd dimensions
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        
        # Apply rotation
        rotated = torch.cat([
            x1 * cos[..., ::2] - x2 * sin[..., ::2],
            x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
        ], dim=-1)
        
        return rotated


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE and optional causal masking"""
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.1,
                 use_bias: bool = False,
                 max_seq_len: int = 8192):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)
        
        # RoPE for positional encoding
        self.rope = RoPEEmbedding(self.head_dim, max_seq_len)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attn_mask: Attention mask [seq_len, seq_len] (for causal masking)
            key_padding_mask: Padding mask [batch, seq_len] (True = ignore)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Shape: [batch, n_heads, seq_len, head_dim]
        
        # Apply RoPE to Q and K
        q, k = self.rope(q, k)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Shape: [batch, n_heads, seq_len, seq_len]
        
        # Apply attention mask (for causal attention)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply key padding mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        # Shape: [batch, n_heads, seq_len, head_dim]
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final projection
        out = self.out_proj(out)
        
        return out