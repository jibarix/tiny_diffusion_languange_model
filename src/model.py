"""
Complete Model Architecture for Tiny Masked Diffusion Language Model

Implements bidirectional transformer with masked diffusion objective:
- Multi-head attention with RoPE positional embeddings
- SwiGLU feedforward networks
- RMSNorm without bias terms
- Masked diffusion training and inference

Architecture follows 2025 research: deeper not wider, compressed vocab.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - CUDA Safe Version"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute frequency tensor - ensure even dimension
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        # x: [batch_size, num_heads, seq_len, head_dim]
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # Clamp sequence length to prevent index errors
        seq_len = min(seq_len, self.max_position_embeddings)
        
        # Generate position encodings
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        
        # Create cos and sin embeddings with proper dimensions
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
        
        # Truncate input if needed
        if x.size(-2) > seq_len:
            x = x[:, :, :seq_len, :]
        
        # Apply rotary embedding
        return self._apply_rotary_pos_emb(x, cos, sin)
    
    def _apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # Ensure head_dim is even for proper splitting
        head_dim = x.size(-1)
        if head_dim % 2 != 0:
            # Pad to make even if necessary
            x = F.pad(x, (0, 1))
            head_dim += 1
        
        # Split x into two halves along the head dimension
        x1, x2 = x.chunk(2, dim=-1)
        
        # Ensure cos/sin match x sequence dimension
        seq_len_x = x.size(-2)
        if cos.size(-2) != seq_len_x:
            cos = cos[:, :, :seq_len_x, :]
            sin = sin[:, :, :seq_len_x, :]
        
        # Expand cos/sin to match x dimensions
        cos = cos.expand_as(x1)  # [batch_size, num_heads, seq_len, head_dim//2]
        sin = sin.expand_as(x1)  # [batch_size, num_heads, seq_len, head_dim//2]
        
        # Apply rotation
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        # Remove padding if it was added
        if x.size(-1) != rotated.size(-1):
            rotated = rotated[..., :x.size(-1)]
        
        return rotated


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE and bidirectional support"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config['d_model']
        self.num_heads = config['n_heads']
        self.head_dim = config['head_dim']
        self.attention_dropout = config.get('attention_dropout', 0.0)
        self.use_causal_mask = config.get('use_causal_mask', False)  # False for diffusion
        
        assert self.hidden_size % self.num_heads == 0
        
        # Linear projections (no bias)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            config.get('max_position_embeddings', 2048),
            config.get('rope_theta', 10000.0)
        )
        
        # Dropout
        self.attention_dropout_layer = nn.Dropout(self.attention_dropout)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        query_states = self.rotary_emb(query_states, seq_len)
        key_states = self.rotary_emb(key_states, seq_len)
        
        # Handle KV cache for inference
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        if use_cache:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        
        # Apply causal mask if needed (disabled for diffusion)
        if self.use_causal_mask:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            causal_mask = causal_mask.to(attn_weights.device)
            attn_weights.masked_fill_(causal_mask, float('-inf'))
        
        # Apply attention mask
        if attention_mask is not None:
            # Convert attention_mask to broadcastable shape
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            
            # Apply mask (0 means mask out)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout_layer(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, present_key_value


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config['d_model']
        self.intermediate_size = config['ffn_hidden_size']
        
        # Three linear layers for SwiGLU
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        # SwiGLU: swish(gate) * up
        gate = F.silu(self.gate_proj(x))  # SiLU is swish
        up = self.up_proj(x)
        intermediate = gate * up
        return self.down_proj(intermediate)


class TransformerBlock(nn.Module):
    """Single transformer block with attention and MLP"""
    
    def __init__(self, config: Dict[str, Any], layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config['d_model']
        self.hidden_dropout = config.get('hidden_dropout', 0.1)
        
        # Attention
        self.self_attn = MultiHeadAttention(config)
        self.input_layernorm = RMSNorm(self.hidden_size, config.get('norm_eps', 1e-6))
        
        # MLP
        self.mlp = SwiGLU(config)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, config.get('norm_eps', 1e-6))
        
        # Dropout
        self.dropout = nn.Dropout(self.hidden_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        # Pre-norm for attention
        normed_hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        attn_output, attn_weights, present_key_value = self.self_attn(
            normed_hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        
        # Residual connection and dropout
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # Pre-norm for MLP
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP
        mlp_output = self.mlp(normed_hidden_states)
        
        # Residual connection and dropout
        hidden_states = hidden_states + self.dropout(mlp_output)
        
        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs


class MaskedDiffusionLM(nn.Module):
    """
    Complete Masked Diffusion Language Model
    
    Bidirectional transformer trained on masked token prediction with
    curriculum learning support for 3-stage training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Model parameters
        self.vocab_size = config['vocab_size']
        self.hidden_size = config['d_model']
        self.num_layers = config['n_layers']
        self.num_heads = config['n_heads']
        self.max_position_embeddings = config.get('max_position_embeddings', 2048)
        self.gradient_checkpointing = config.get('gradient_checkpointing', False)
        
        # Special token IDs (set by tokenizer)
        self.pad_token_id = config.get('pad_token_id')
        self.mask_token_id = config.get('mask_token_id')
        
        # Embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx) 
            for layer_idx in range(self.num_layers)
        ])
        
        # Final norm and head
        self.norm = RMSNorm(self.hidden_size, config.get('norm_eps', 1e-6))
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Tie embeddings to output weights (standard practice)
        self.tie_weights()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def tie_weights(self):
        """Tie input embeddings to output projection"""
        self.lm_head.weight = self.embed_tokens.weight
    
    def _init_weights(self, module):
        """Initialize weights following standard practice"""
        std = self.config.get('initializer_range', 0.02)
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.get('use_cache', False)
        return_dict = return_dict if return_dict is not None else True
        
        # Input validation
        if input_ids is None:
            raise ValueError("input_ids cannot be None")
        
        batch_size, seq_length = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=input_ids.device)
        
        # Input embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare for caching
        if use_cache:
            use_cache = True
            if past_key_values is None:
                past_key_values = [None] * len(self.layers)
        else:
            past_key_values = [None] * len(self.layers)
        
        # Forward through transformer layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        for idx, (decoder_layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache=False)
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    None,  # past_key_value
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
            
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Flatten for cross-entropy
            shift_logits = logits.view(-1, self.vocab_size)
            shift_labels = labels.view(-1)
            
            # Compute loss only on non-ignored positions (labels != -100)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            return output + (hidden_states, all_self_attentions, all_hidden_states)
        
        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': next_decoder_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attentions,
        }
    
    def generate_step(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> torch.LongTensor:
        """
        Generate one step of masked diffusion.
        
        For inference, this predicts tokens for all masked positions.
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            
            logits = outputs['logits']  # [batch_size, seq_len, vocab_size]
            
            # Find masked positions
            if self.mask_token_id is not None:
                mask_positions = (input_ids == self.mask_token_id)
            else:
                # Fallback: assume token_id 1 is mask
                mask_positions = (input_ids == 1)
            
            # Sample from masked positions
            next_token_ids = input_ids.clone()
            
            for batch_idx in range(input_ids.size(0)):
                for seq_idx in range(input_ids.size(1)):
                    if mask_positions[batch_idx, seq_idx]:
                        token_logits = logits[batch_idx, seq_idx]
                        
                        # Apply temperature
                        if temperature != 1.0:
                            token_logits = token_logits / temperature
                        
                        # Apply top-k filtering
                        if top_k > 0:
                            top_k_logits, _ = torch.topk(token_logits, min(top_k, token_logits.size(-1)))
                            token_logits[token_logits < top_k_logits[-1]] = float('-inf')
                        
                        # Apply top-p (nucleus) filtering
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            
                            # Remove tokens with cumulative probability above the threshold
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                            sorted_indices_to_remove[0] = 0
                            
                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            token_logits[indices_to_remove] = float('-inf')
                        
                        # Sample or take argmax
                        if do_sample:
                            probs = F.softmax(token_logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.argmax(token_logits, dim=-1, keepdim=True)
                        
                        next_token_ids[batch_idx, seq_idx] = next_token
            
            return next_token_ids
    
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        num_diffusion_steps: int = 20,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Generate text using masked diffusion process.
        
        This implements the full diffusion sampling process:
        1. Start with all masks
        2. Iteratively unmask tokens over multiple steps
        3. Return final generated sequence
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided for generation")
        
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = self.pad_token_id or 0
        if eos_token_id is None:
            eos_token_id = pad_token_id
        
        # Initialize with all masks for new tokens
        mask_token_id = self.mask_token_id or 1
        
        # Extend input with masked tokens
        new_tokens = torch.full((batch_size, max_new_tokens), mask_token_id, device=device, dtype=torch.long)
        
        # Combine prompt with masked new tokens
        current_ids = torch.cat([input_ids, new_tokens], dim=1)
        
        # Create attention mask
        if attention_mask is not None:
            new_attention_mask = torch.ones((batch_size, max_new_tokens), device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_attention_mask], dim=1)
        else:
            attention_mask = torch.ones_like(current_ids)
        
        # Diffusion sampling loop
        for step in range(num_diffusion_steps):
            # Generate next step
            current_ids = self.generate_step(
                input_ids=current_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )
        
        return current_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params
    
    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model flops utilization (MFU)"""
        # First estimate the number of flops we do per iteration
        # See PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg['n_layers'], cfg['n_heads'], cfg['d_model']//cfg['n_heads'], cfg.get('sequence_length', 512)
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # Express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu


def create_model_from_config(config: Dict[str, Any]) -> MaskedDiffusionLM:
    """Create model instance from configuration"""
    # Validate configuration
    required_keys = ['d_model', 'n_layers', 'n_heads', 'vocab_size']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Create model
    model = MaskedDiffusionLM(config)
    
    # Print model info
    num_params = model.get_num_params()
    print(f"Model created with {num_params:,} parameters")
    
    return model


# Model loading and saving utilities
def save_model_checkpoint(
    model: MaskedDiffusionLM,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    epoch: int,
    step: int,
    loss: float,
    filepath: str
):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'epoch': epoch,
        'step': step,
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_model_checkpoint(filepath: str, device: str = 'cpu') -> Tuple[MaskedDiffusionLM, Dict[str, Any]]:
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create model from config
    config = checkpoint['config']
    model = create_model_from_config(config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Checkpoint loaded: {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Step: {checkpoint.get('step', 'unknown')}")
    print(f"  Loss: {checkpoint.get('loss', 'unknown')}")
    
    return model, checkpoint


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    test_config = {
        'd_model': 128,
        'n_layers': 3,
        'n_heads': 4,
        'head_dim': 32,
        'ffn_hidden_size': 320,
        'vocab_size': 5000,
        'max_position_embeddings': 512,
        'attention_dropout': 0.0,
        'hidden_dropout': 0.1,
        'use_causal_mask': False,  # Bidirectional for diffusion
        'gradient_checkpointing': False,
        'norm_eps': 1e-6,
        'initializer_range': 0.02,
        'mask_token_id': 1,
        'pad_token_id': 0,
    }
    
    print("Testing model architecture...")
    
    # Create model
    model = create_model_from_config(test_config)
    model.eval()
    
    # Test input
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, test_config['vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :seq_len//2] = -100  # Mask first half for loss
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss']:.4f}" if outputs['loss'] is not None else "No loss")
    
    # Test generation
    prompt = torch.randint(0, test_config['vocab_size'], (1, 10))
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt,
            max_new_tokens=20,
            num_diffusion_steps=5,
            do_sample=False,
        )
    
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")
    
    # Memory usage
    num_params = model.get_num_params()
    model_size_mb = num_params * 4 / 1024 / 1024  # 4 bytes per param (fp32)
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {model_size_mb:.1f} MB")
    
    print("Model architecture test complete!")