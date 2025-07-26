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
import random
import logging # --- ADDED: Import the logging module ---
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
    """Rotary Position Embedding (RoPE) - Production Implementation"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cos/sin cache if needed"""
        if seq_len > self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device:
            # Limit to max_position_embeddings
            seq_len = min(seq_len, self.max_position_embeddings)
            
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            
            # Create embeddings [seq_len, dim//2]
            cos = freqs.cos().to(dtype)
            sin = freqs.sin().to(dtype)
            
            self._cos_cached = cos
            self._sin_cached = sin

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        # x shape: [batch_size, num_heads, seq_len, head_dim]
        batch_size, num_heads, x_seq_len, head_dim = x.shape
        
        if seq_len is None:
            seq_len = x_seq_len
        
        # Ensure we don't exceed limits
        seq_len = min(seq_len, x_seq_len, self.max_position_embeddings)
        
        # Truncate input if needed
        if x_seq_len > seq_len:
            x = x[:, :, :seq_len, :]
        
        # Update cache
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        
        # Get cached values
        cos = self._cos_cached[:seq_len]  # [seq_len, dim//2]
        sin = self._sin_cached[:seq_len]  # [seq_len, head_dim//2]
        
        # Apply rotation
        return self._apply_rotary_emb(x, cos, sin)
    
    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Apply rotary embedding to input tensor"""
        # x: [batch_size, num_heads, seq_len, head_dim]
        # cos, sin: [seq_len, head_dim//2]
        
        # Handle odd dimensions
        if x.size(-1) % 2 != 0:
            # Pad last dimension to make even
            x = F.pad(x, (0, 1))
        
        # Reshape for rotation: split into pairs
        x_reshaped = x.view(*x.shape[:-1], -1, 2)  # [..., head_dim//2, 2]
        x1, x2 = x_reshaped.unbind(dim=-1)  # [..., head_dim//2]
        
        # Expand cos/sin to match input dimensions
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Recombine
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        rotated = rotated.view(*x.shape[:-1], -1)  # Back to original shape
        
        # Remove padding if it was added
        if rotated.size(-1) > x.size(-1):
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
        output_attentions: bool = False,
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
        if output_attentions:
            outputs += (attn_weights,)
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
        
        # CRITICAL FIX: Proper special token ID mapping
        # Based on the new tokenizer order: 0=EOS, 1=MASK, 2=PAD
        self.eos_token_id = config.get('eos_token_id', 0)  # <|endoftext|> at position 0
        self.mask_token_id = config.get('mask_token_id', 1)  # [MASK] at position 1
        self.pad_token_id = config.get('pad_token_id', 2)   # [PAD] at position 2 (NOT 0!)
        
        # Log the token mapping for verification
        print(f"Model token configuration:")
        print(f"  EOS token ID: {self.eos_token_id}")
        print(f"  MASK token ID: {self.mask_token_id}")
        print(f"  PAD token ID: {self.pad_token_id}")
        
        # Set padding_idx in the embedding layer
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_token_id)
        
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
    
    # --- NEW: Noise schedule for MDLM objective ---
    # Implements the log-linear noise schedule from Sahoo et al.
    def get_log_linear_schedule(self, t):
        """
        Returns alpha_t from a log-linear noise schedule.
        alpha_t is the probability of a token remaining unmasked.
        """
        return torch.exp(-torch.log(1 - t.clamp(min=0, max=0.999)))

    def get_alpha_t_and_weight(self, t, device):
        """
        Calculates alpha_t and the loss weight for a given timestep t.
        """
        # Ensure t is a tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=device)
        
        t = t.to(device)
        
        # Define alpha_t using the log-linear schedule
        alpha_t = 1.0 - t
        alpha_t_prime = -1.0 # Derivative of (1-t) w.r.t t
        
        # Loss weight is -alpha_t' / (1 - alpha_t)
        weight = -alpha_t_prime / (1.0 - alpha_t)
        
        return alpha_t, weight
    # --- END NEW ---

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
            label_smoothing: Optional[float] = None,
        ):
            # --- MODIFIED: Conditional logic for training vs. inference ---
            # If `labels` are provided, we are in training mode and apply the MDLM objective.
            # If `labels` are None, we are in inference mode and run a standard forward pass.
            
            use_cache = use_cache if use_cache is not None else self.config.get('use_cache', False)
            return_dict = return_dict if return_dict is not None else True
            
            if input_ids is None:
                raise ValueError("input_ids cannot be None")

            # --- BRANCH 1: TRAINING LOGIC (MDLM Objective) ---
            if labels is not None:
                batch_size, seq_length = input_ids.shape
                device = input_ids.device
                clean_labels = labels.clone() # `labels` are the clean tokens

                # 1. Sample a random timestep t for each sequence in the batch
                t_start = torch.arange(batch_size, device=device) / batch_size
                t_end = torch.arange(1, batch_size + 1, device=device) / batch_size
                t = torch.rand(batch_size, device=device) * (t_end - t_start) + t_start
                t = t.view(-1, 1)

                # --- MODIFIED: Changed print() to logging.info() ---
                if self.training and random.random() < 0.01:
                    logging.info(f"[CONSOLE LOG] MDLM Objective: Sampled t values (min/max): {t.min().item():.3f}/{t.max().item():.3f}")
                # --- END MODIFIED ---

                # 2. Get noise schedule value alpha_t and loss weight
                alpha_t, weight = self.get_alpha_t_and_weight(t, device)

                # 3. Create the masked input z_t
                mask_prob = 1.0 - alpha_t
                rand_mask = torch.rand(batch_size, seq_length, device=device) < mask_prob
                if attention_mask is not None:
                    rand_mask = rand_mask & attention_mask.bool()
                
                masked_input_ids = torch.where(rand_mask, self.mask_token_id, clean_labels)
                
                # 4. Standard transformer forward pass on the masked input
                hidden_states = self.embed_tokens(masked_input_ids)
            # --- END TRAINING-SPECIFIC MASKING ---
            
            # --- BRANCH 2: INFERENCE LOGIC ---
            else:
                # In inference, input_ids are already masked by the generate function.
                # We just need to pass them through the model.
                hidden_states = self.embed_tokens(input_ids)
            # --- END INFERENCE LOGIC ---

            # --- COMMON TRANSFORMER FORWARD PASS ---
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None
            
            for decoder_layer in self.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                )
                hidden_states = layer_outputs[0]
            
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            # --- END COMMON TRANSFORMER FORWARD PASS ---

            loss = None
            # --- LOSS CALCULATION (only in training) ---
            if labels is not None:
                # --- BUG FIX: Stabilized loss calculation ---
                # The previous weighting scheme (1/t) was numerically unstable and caused
                # the loss to explode. This version computes a simple, unweighted
                # cross-entropy loss, which is far more stable for training.
                loss_fct = CrossEntropyLoss(ignore_index=self.pad_token_id)
                
                # Calculate the loss only on the tokens that were actually masked
                loss_mask = rand_mask.view(-1)
                
                # Filter logits and labels to only include masked positions
                masked_logits = logits.view(-1, self.vocab_size)[loss_mask]
                masked_labels = labels.view(-1)[loss_mask]
                
                if masked_labels.numel() > 0:
                    # Calculate a simple, unweighted cross-entropy loss
                    loss = loss_fct(masked_logits, masked_labels)
                else:
                    # If no tokens were masked in the batch, loss is 0
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                # --- END BUG FIX ---

            if not return_dict:
                return (loss, logits) if loss is not None else (logits,)

            return {
                'loss': loss,
                'logits': logits,
                'past_key_values': None,
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
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None, # Set labels to None during inference
                use_cache=False,
            )
            
            logits = outputs['logits']  # [batch_size, seq_len, vocab_size]
            
            if self.mask_token_id is not None:
                mask_positions = (input_ids == self.mask_token_id)
            else:
                mask_positions = (input_ids == 1)
            
            next_token_ids = input_ids.clone()
            
            for batch_idx in range(input_ids.size(0)):
                for seq_idx in range(input_ids.size(1)):
                    if mask_positions[batch_idx, seq_idx]:
                        token_logits = logits[batch_idx, seq_idx]
                        
                        # --- FIX: Prevent generation of special tokens ---
                        # Set logits for PAD, EOS, and MASK tokens to -inf so they are not sampled.
                        token_logits[self.pad_token_id] = -float('inf')
                        token_logits[self.eos_token_id] = -float('inf')
                        token_logits[self.mask_token_id] = -float('inf')
                        # --- END FIX ---
                        
                        if temperature != 1.0:
                            token_logits = token_logits / temperature
                        
                        if top_k > 0:
                            top_k_logits, _ = torch.topk(token_logits, min(top_k, token_logits.size(-1)))
                            token_logits[token_logits < top_k_logits[-1]] = float('-inf')
                        
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                            sorted_indices_to_remove[0] = 0
                            
                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            token_logits[indices_to_remove] = float('-inf')
                        
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
        temperature: float = 0.6,
        top_p: float = 0.85,
        top_k: int = 20,
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
        
        # Use model's configured token IDs as robust defaults
        pad_token_id = self.pad_token_id if pad_token_id is None else pad_token_id
        eos_token_id = self.eos_token_id if eos_token_id is None else eos_token_id
        mask_token_id = self.mask_token_id

        if mask_token_id is None:
            raise ValueError("Mask token ID is not configured in the model.")
        
        # Initialize with all masks for new tokens
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

    # --- NEW: Semi-Autoregressive (SAR) Sampling Method ---
    # Implements the efficient SAR decoding from Sahoo et al. to generate arbitrary length text.
    def generate_sar(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 2048,
        generation_block_size: int = 512,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generate text of arbitrary length using Semi-Autoregressive (SAR) decoding.
        """
        print(f"[CONSOLE LOG] Starting Semi-Autoregressive generation to target length {max_length}.")

        if input_ids.shape[1] >= max_length:
            return input_ids

        generated_sequence = input_ids
        prompt_length = input_ids.shape[1]

        while generated_sequence.shape[1] < max_length:
            current_length = generated_sequence.shape[1]
            
            # Determine how many new tokens to generate in this block
            remaining_tokens = max_length - current_length
            tokens_to_generate = min(generation_block_size, remaining_tokens)
            
            # The context is the end of the current sequence
            context = generated_sequence
            
            print(f"[CONSOLE LOG] SAR Step: current_len={current_length}, generating {tokens_to_generate} new tokens.")

            # Call the standard generate method with the context and new masks
            block_output = self.generate(
                input_ids=context,
                max_new_tokens=tokens_to_generate,
                **kwargs
            )
            
            # Extract only the newly generated tokens
            newly_generated_tokens = block_output[:, current_length:]
            
            # Append the new tokens to our sequence
            generated_sequence = torch.cat([generated_sequence, newly_generated_tokens], dim=1)

        print("[CONSOLE LOG] SAR generation complete.")
        return generated_sequence
    # --- END NEW ---
    
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
    
    # *** FIX: Pass the 'model' sub-dictionary from the project config ***
    # The create_model_from_config function expects the model-specific config, not the entire project config.
    model_config = config.get('model', config) # Use .get for backward compatibility if 'model' key is missing
    model = create_model_from_config(model_config)
    
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
