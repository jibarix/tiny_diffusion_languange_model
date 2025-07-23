"""
Multi-Head Attention with RoPE - REVISED WITH CONFIG SUPPORT
Bidirectional attention for diffusion (no causal masking)
All hardcoded parameters moved to configuration
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RoPEEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) with configurable parameters"""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base  # Now properly configurable
        
        # Precompute frequency tensor using configurable base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for efficiency
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0
        self._cached_device = None
    
    def _get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached or compute cos/sin for given sequence length"""
        # Check if we need to recompute cache
        need_recompute = (
            seq_len > self._cached_seq_len or 
            self._cached_cos is None or 
            self._cached_device != device
        )
        
        if need_recompute:
            # Compute new cache
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cached_cos = emb.cos()
            self._cached_sin = emb.sin()
            self._cached_seq_len = seq_len
            self._cached_device = device
        
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors"""
        seq_len = q.size(-2)
        cos, sin = self._get_cos_sin(seq_len, q.device)
        
        return self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)
    
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to tensor
        
        Args:
            x: Input tensor [..., seq_len, dim]
            cos: Cosine values [seq_len, dim]
            sin: Sine values [seq_len, dim]
        """
        # Ensure cos and sin have the right dimensions
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        if sin.dim() == 2:
            sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        
        # Split into even/odd dimensions
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        
        # Apply rotation
        cos_part = cos[..., ::2] if cos.size(-1) == x.size(-1) else cos[..., :x1.size(-1)]
        sin_part = sin[..., 1::2] if sin.size(-1) == x.size(-1) else sin[..., :x2.size(-1)]
        
        rotated = torch.stack([
            x1 * cos_part - x2 * sin_part,
            x1 * sin_part + x2 * cos_part
        ], dim=-1)
        
        # Flatten the last two dimensions back
        return rotated.flatten(-2)
    
    def get_config_dict(self) -> dict:
        """Get configuration dictionary for saving/loading"""
        return {
            'dim': self.dim,
            'max_seq_len': self.max_seq_len,
            'base': self.base
        }


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE and configurable parameters"""
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.1,
                 use_bias: bool = False,
                 max_seq_len: int = 8192,
                 rope_base: int = 10000,        # NEW: Configurable RoPE base
                 attention_scale: Optional[float] = None,  # NEW: Configurable attention scale
                 qk_norm: bool = False):        # NEW: Query-Key normalization option
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Configurable attention scale
        self.scale = attention_scale if attention_scale is not None else 1.0 / math.sqrt(self.head_dim)
        
        # Store configuration
        self.use_bias = use_bias
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self.qk_norm = qk_norm
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)
        
        # RoPE for positional encoding with configurable base
        self.rope = RoPEEmbedding(self.head_dim, max_seq_len, rope_base)
        
        # Optional Query-Key normalization (used in some recent models)
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
            self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # For attention weight extraction (useful for analysis)
        self.last_attention_weights = None
        
    def forward(self, 
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attn_mask: Attention mask [seq_len, seq_len] (for causal masking)
            key_padding_mask: Padding mask [batch, seq_len] (True = ignore)
            return_attention_weights: Whether to return attention weights
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
        
        # Apply Query-Key normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Apply RoPE to Q and K
        q, k = self.rope(q, k)
        
        # Compute attention with optional return of weights
        out, attn_weights = self._compute_attention(
            q, k, v, attn_mask, key_padding_mask, return_weights=return_attention_weights
        )
        
        # Store attention weights for analysis
        if return_attention_weights:
            self.last_attention_weights = attn_weights
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final projection
        out = self.out_proj(out)
        
        if return_attention_weights:
            return out, attn_weights
        return out
    
    def _compute_attention(self, 
                          q: torch.Tensor, 
                          k: torch.Tensor, 
                          v: torch.Tensor,
                          attn_mask: Optional[torch.Tensor] = None,
                          key_padding_mask: Optional[torch.Tensor] = None,
                          return_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute attention scores and apply to values"""
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Shape: [batch, n_heads, seq_len, seq_len]
        
        # Apply attention mask (for causal attention)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # Expand for batch and head dimensions
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [batch, seq_len], True = mask (ignore)
            # Expand for head and query dimensions
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Softmax with numerical stability
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        # Shape: [batch, n_heads, seq_len, head_dim]
        
        if return_weights:
            return out, attn_weights
        return out, None
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get the last computed attention weights"""
        return self.last_attention_weights
    
    def clear_attention_weights(self):
        """Clear stored attention weights to save memory"""
        self.last_attention_weights = None
    
    def get_config_dict(self) -> dict:
        """Get configuration dictionary for saving/loading"""
        return {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'head_dim': self.head_dim,
            'scale': self.scale,
            'use_bias': self.use_bias,
            'max_seq_len': self.max_seq_len,
            'rope_base': self.rope_base,
            'qk_norm': self.qk_norm,
            'rope_config': self.rope.get_config_dict()
        }
    
    def compute_attention_stats(self, x: torch.Tensor) -> dict:
        """Compute attention statistics for analysis"""
        with torch.no_grad():
            batch_size, seq_len, _ = x.shape
            
            # Forward pass to get attention weights
            _, attn_weights = self.forward(x, return_attention_weights=True)
            
            if attn_weights is None:
                return {}
            
            # Compute statistics
            stats = {
                'attention_entropy': self._compute_attention_entropy(attn_weights),
                'attention_sparsity': self._compute_attention_sparsity(attn_weights),
                'head_diversity': self._compute_head_diversity(attn_weights),
                'positional_bias': self._compute_positional_bias(attn_weights)
            }
            
            return stats
    
    def _compute_attention_entropy(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention weights"""
        # attn_weights: [batch, n_heads, seq_len, seq_len]
        log_attn = torch.log(attn_weights + 1e-8)
        entropy = -(attn_weights * log_attn).sum(dim=-1)  # [batch, n_heads, seq_len]
        return entropy.mean()
    
    def _compute_attention_sparsity(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Compute sparsity of attention weights (fraction of near-zero weights)"""
        threshold = 0.01
        sparse_mask = attn_weights < threshold
        sparsity = sparse_mask.float().mean()
        return sparsity
    
    def _compute_head_diversity(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Compute diversity between attention heads"""
        # Flatten spatial dimensions for correlation computation
        batch_size, n_heads, seq_len, _ = attn_weights.shape
        flat_attn = attn_weights.view(batch_size, n_heads, -1)  # [batch, n_heads, seq_len^2]
        
        # Compute pairwise correlations between heads
        correlations = []
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                corr = F.cosine_similarity(flat_attn[:, i], flat_attn[:, j], dim=-1)
                correlations.append(corr.mean())
        
        if correlations:
            # Diversity = 1 - average correlation
            avg_correlation = torch.stack(correlations).mean()
            return 1.0 - avg_correlation
        return torch.tensor(0.0)
    
    def _compute_positional_bias(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Compute positional bias in attention (preference for certain positions)"""
        # Average attention weights across batch and heads
        avg_attn = attn_weights.mean(dim=(0, 1))  # [seq_len, seq_len]
        
        # Compute bias toward diagonal (local attention)
        seq_len = avg_attn.size(0)
        diagonal_mask = torch.eye(seq_len, device=avg_attn.device)
        diagonal_attention = (avg_attn * diagonal_mask).sum()
        
        # Normalize by sequence length
        diagonal_bias = diagonal_attention / seq_len
        
        return diagonal_bias
    
    def visualize_attention_pattern(self, x: torch.Tensor, head_idx: int = 0, 
                                   save_path: Optional[str] = None) -> torch.Tensor:
        """Visualize attention pattern for a specific head"""
        with torch.no_grad():
            _, attn_weights = self.forward(x, return_attention_weights=True)
            
            if attn_weights is None:
                raise ValueError("No attention weights available")
            
            # Get attention pattern for specified head (first batch item)
            pattern = attn_weights[0, head_idx]  # [seq_len, seq_len]
            
            if save_path:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                plt.imshow(pattern.cpu().numpy(), cmap='Blues', aspect='auto')
                plt.colorbar()
                plt.title(f'Attention Pattern - Head {head_idx}')
                plt.xlabel('Key Position')
                plt.ylabel('Query Position')
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Attention pattern saved to {save_path}")
            
            return pattern
    
    def get_memory_footprint(self) -> dict:
        """Get estimated memory footprint of attention module"""
        def get_param_size(module):
            return sum(p.numel() * 4 for p in module.parameters()) / 1024**2  # MB
        
        return {
            'q_proj_mb': get_param_size(self.q_proj),
            'k_proj_mb': get_param_size(self.k_proj),
            'v_proj_mb': get_param_size(self.v_proj),
            'out_proj_mb': get_param_size(self.out_proj),
            'total_attention_mb': get_param_size(self),
            'cache_overhead_mb': self._estimate_cache_overhead()
        }
    
    def _estimate_cache_overhead(self) -> float:
        """Estimate memory overhead from RoPE caching"""
        if self.rope._cached_cos is not None:
            cos_size = self.rope._cached_cos.numel() * 4 / 1024**2  # MB
            sin_size = self.rope._cached_sin.numel() * 4 / 1024**2  # MB
            return cos_size + sin_size
        return 0.0
    
    def reset_parameters(self, init_std: float = 0.02):
        """Reset all parameters with specified initialization"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        if self.qk_norm:
            # LayerNorm parameters are initialized by default
            pass