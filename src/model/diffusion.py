"""
Masked Diffusion Language Model - REVISED WITH CONFIG SUPPORT
Orchestrates transformer for diffusion training and inference
All hardcoded parameters moved to configuration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from .transformer import TransformerModel


class MaskedDiffusionLM(nn.Module):
    """Masked Diffusion Language Model with full configuration support"""
    
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
                 mask_token_id: int = 1,
                 init_std: float = 0.02,              # NEW: Configurable initialization
                 rope_base: int = 10000,              # NEW: Configurable RoPE base
                 attention_scale: Optional[float] = None,  # NEW: Configurable attention scale
                 qk_norm: bool = False,               # NEW: Query-Key normalization
                 min_masks_per_sample: int = 1,       # NEW: Configurable minimum masks
                 max_masking_rate: float = 0.95,      # NEW: Configurable maximum masking
                 generation_temperature: float = 1.0, # NEW: Default generation temperature
                 generation_top_k: Optional[int] = None,  # NEW: Default top-k
                 generation_top_p: Optional[float] = None):  # NEW: Default top-p
        super().__init__()
        
        # Core model parameters
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Token IDs
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        
        # Configuration parameters
        self.init_std = init_std
        self.rope_base = rope_base
        self.attention_scale = attention_scale
        self.qk_norm = qk_norm
        self.min_masks_per_sample = min_masks_per_sample
        self.max_masking_rate = max_masking_rate
        
        # Generation defaults
        self.generation_temperature = generation_temperature
        self.generation_top_k = generation_top_k
        self.generation_top_p = generation_top_p
        
        # Transformer backbone with full configuration support
        self.transformer = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_bias=use_bias,
            norm_eps=norm_eps,
            pad_token_id=pad_token_id,
            init_std=init_std,          # NEW: Pass configurable init_std
            rope_base=rope_base         # NEW: Pass configurable rope_base
        )
        
        # Training state tracking
        self.training_step = 0
        self.last_loss = None
        self.corruption_stats = {
            'total_masks': 0,
            'total_tokens': 0,
            'avg_masking_rate': 0.0
        }
        
    def create_corruption_mask(self, 
                              input_ids: torch.Tensor,
                              masking_rate: float,
                              min_masks: Optional[int] = None,
                              mask_strategy: str = "random") -> torch.Tensor:
        """Create corruption mask with configurable strategies"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Use configured minimum masks if not specified
        if min_masks is None:
            min_masks = self.min_masks_per_sample
        
        # Clamp masking rate to configured maximum
        masking_rate = min(masking_rate, self.max_masking_rate)
        
        # Don't mask padding tokens
        valid_positions = (input_ids != self.pad_token_id)
        
        masks = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for i in range(batch_size):
            valid_pos = valid_positions[i].nonzero(as_tuple=True)[0]
            if len(valid_pos) == 0:
                continue
                
            n_valid = len(valid_pos)
            n_mask = max(min_masks, int(n_valid * masking_rate))
            n_mask = min(n_mask, n_valid)  # Don't exceed valid positions
            
            if mask_strategy == "random":
                # Standard random masking
                mask_indices = torch.randperm(n_valid, device=device)[:n_mask]
                masks[i, valid_pos[mask_indices]] = True
                
            elif mask_strategy == "span":
                # Span masking (mask contiguous tokens)
                masks[i] = self._create_span_mask(valid_pos, n_mask, device)
                
            elif mask_strategy == "block":
                # Block masking (mask sentence-like blocks)
                masks[i] = self._create_block_mask(input_ids[i], valid_pos, n_mask, device)
                
            else:
                raise ValueError(f"Unknown mask_strategy: {mask_strategy}")
        
        # Update corruption statistics
        self._update_corruption_stats(masks, valid_positions)
        
        return masks
    
    def _create_span_mask(self, valid_pos: torch.Tensor, n_mask: int, device: torch.device) -> torch.Tensor:
        """Create span-based corruption mask"""
        mask = torch.zeros(len(valid_pos), dtype=torch.bool, device=device)
        
        # Average span length (configurable in future versions)
        avg_span_length = 3
        
        masked_count = 0
        while masked_count < n_mask:
            # Choose random start position
            start_idx = torch.randint(0, len(valid_pos), (1,), device=device).item()
            
            # Choose span length
            span_length = max(1, torch.poisson(torch.tensor(avg_span_length - 1, device=device)).item() + 1)
            span_length = min(span_length, n_mask - masked_count)
            
            # Apply span mask
            end_idx = min(start_idx + span_length, len(valid_pos))
            mask[start_idx:end_idx] = True
            masked_count += (end_idx - start_idx)
        
        return mask
    
    def _create_block_mask(self, input_ids: torch.Tensor, valid_pos: torch.Tensor, 
                          n_mask: int, device: torch.device) -> torch.Tensor:
        """Create block-based corruption mask (mask sentence-like units)"""
        # Simple heuristic: split on punctuation
        mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
        
        # For now, fall back to random masking
        # In future versions, this could use punctuation detection
        mask_indices = torch.randperm(len(valid_pos), device=device)[:n_mask]
        mask[valid_pos[mask_indices]] = True
        
        return mask
    
    def _update_corruption_stats(self, masks: torch.Tensor, valid_positions: torch.Tensor):
        """Update corruption statistics for monitoring"""
        total_masks = masks.sum().item()
        total_valid = valid_positions.sum().item()
        
        self.corruption_stats['total_masks'] += total_masks
        self.corruption_stats['total_tokens'] += total_valid
        
        if total_valid > 0:
            current_rate = total_masks / total_valid
            # Exponential moving average
            alpha = 0.1
            self.corruption_stats['avg_masking_rate'] = (
                alpha * current_rate + (1 - alpha) * self.corruption_stats['avg_masking_rate']
            )
    
    def apply_corruption(self,
                        input_ids: torch.Tensor,
                        corruption_mask: torch.Tensor) -> torch.Tensor:
        """Apply masking corruption to input"""
        corrupted_ids = input_ids.clone()
        corrupted_ids[corruption_mask] = self.mask_token_id
        return corrupted_ids
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                masking_rate: Optional[float] = None,
                corruption_mask: Optional[torch.Tensor] = None,
                mask_strategy: str = "random",
                return_dict: bool = True) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass for training or inference
        
        Args:
            input_ids: [batch, seq_len] - clean input tokens
            attention_mask: [batch, seq_len] - attention mask
            masking_rate: float - random masking rate (for training)
            corruption_mask: [batch, seq_len] - explicit mask (overrides masking_rate)
            mask_strategy: str - masking strategy ("random", "span", "block")
            return_dict: bool - whether to return dictionary or just logits
        """
        
        # Training mode: apply random corruption
        if self.training and masking_rate is not None:
            corruption_mask = self.create_corruption_mask(
                input_ids, masking_rate, mask_strategy=mask_strategy
            )
            corrupted_input = self.apply_corruption(input_ids, corruption_mask)
            targets = input_ids
            
        # Inference mode: input is already corrupted
        elif corruption_mask is not None:
            corrupted_input = self.apply_corruption(input_ids, corruption_mask)
            targets = input_ids
            
        else:
            # No corruption (e.g., validation without masking)
            corrupted_input = input_ids
            targets = input_ids
            corruption_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Forward through transformer
        logits = self.transformer(
            input_ids=corrupted_input,
            attention_mask=attention_mask,
            causal_mask=False  # Always bidirectional for diffusion
        )
        
        # Update training step counter
        if self.training:
            self.training_step += 1
        
        if return_dict:
            outputs = {
                'logits': logits,
                'targets': targets,
                'corruption_mask': corruption_mask,
                'corrupted_input': corrupted_input,
                'masking_rate': masking_rate,
                'mask_strategy': mask_strategy
            }
            return outputs
        
        return logits
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    loss_type: str = "cross_entropy",
                    label_smoothing: float = 0.0,
                    reduction: str = "mean") -> torch.Tensor:
        """Compute diffusion training loss with configurable options"""
        logits = outputs['logits']  # [batch, seq_len, vocab_size]
        targets = outputs['targets']  # [batch, seq_len]
        corruption_mask = outputs['corruption_mask']  # [batch, seq_len]
        
        # Only compute loss on masked positions
        loss_mask = corruption_mask.view(-1)
        
        if loss_mask.sum() == 0:
            # No masked tokens
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Flatten for loss computation
        flat_logits = logits.view(-1, self.vocab_size)
        flat_targets = targets.view(-1)
        
        # Apply loss mask
        masked_logits = flat_logits[loss_mask]
        masked_targets = flat_targets[loss_mask]
        
        # Compute loss based on type
        if loss_type == "cross_entropy":
            loss = F.cross_entropy(
                masked_logits,
                masked_targets,
                label_smoothing=label_smoothing,
                reduction=reduction
            )
        elif loss_type == "focal":
            # Focal loss for handling difficult examples
            ce_loss = F.cross_entropy(masked_logits, masked_targets, reduction='none')
            pt = torch.exp(-ce_loss)
            alpha, gamma = 1.0, 2.0  # Configurable in future versions
            focal_loss = alpha * (1 - pt) ** gamma * ce_loss
            loss = focal_loss.mean() if reduction == "mean" else focal_loss.sum()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        # Store for monitoring
        self.last_loss = loss.item()
        
        return loss
    
    def generate_step(self,
                     input_ids: torch.Tensor,
                     temperature: Optional[float] = None,
                     top_k: Optional[int] = None,
                     top_p: Optional[float] = None,
                     confidence_threshold: float = 0.0) -> torch.Tensor:
        """Single denoising step during generation with enhanced control"""
        self.eval()
        
        # Use configured defaults if not specified
        temperature = temperature or self.generation_temperature
        top_k = top_k or self.generation_top_k
        top_p = top_p or self.generation_top_p
        
        with torch.no_grad():
            # Find masked positions
            mask_positions = (input_ids == self.mask_token_id)
            
            if not mask_positions.any():
                return input_ids  # No masks to fill
            
            # Forward pass
            outputs = self.forward(input_ids, return_dict=True)
            logits = outputs['logits']  # [batch, seq_len, vocab_size]
            
            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                logits = self._apply_top_k_filtering(logits, top_k)
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                logits = self._apply_top_p_filtering(logits, top_p)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample tokens
            sampled_tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(input_ids.shape)
            
            # Apply confidence-based selective unmasking
            if confidence_threshold > 0.0:
                # Only unmask tokens with high confidence
                max_probs = probs.max(dim=-1)[0]  # [batch, seq_len]
                high_confidence = max_probs > confidence_threshold
                selective_mask = mask_positions & high_confidence
            else:
                selective_mask = mask_positions
            
            # Update only selected masked positions
            new_input_ids = input_ids.clone()
            new_input_ids[selective_mask] = sampled_tokens[selective_mask]
            
            return new_input_ids
    
    def _apply_top_k_filtering(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k filtering to logits"""
        if k <= 0:
            return logits
        
        k = min(k, logits.size(-1))
        top_k_values, _ = torch.topk(logits, k, dim=-1)
        min_values = top_k_values[..., [-1]]
        
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    
    def _apply_top_p_filtering(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering to logits"""
        if p >= 1.0:
            return logits
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Create mask for tokens to remove
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        
        return torch.where(indices_to_remove, torch.full_like(logits, float('-inf')), logits)
    
    def generate(self,
                prompt_ids: torch.Tensor,
                max_length: int,
                num_steps: int = 50,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                confidence_schedule: str = "constant",
                confidence_threshold: float = 0.0) -> torch.Tensor:
        """Generate text using iterative demasking with enhanced control"""
        self.eval()
        
        batch_size = prompt_ids.size(0)
        device = prompt_ids.device
        
        # Start with fully masked sequence (except prompt)
        generated_ids = torch.full((batch_size, max_length), self.mask_token_id, device=device)
        generated_ids[:, :prompt_ids.size(1)] = prompt_ids
        
        # Iteratively demask with optional confidence scheduling
        for step in range(num_steps):
            # Dynamic confidence threshold
            if confidence_schedule == "linear":
                current_confidence = confidence_threshold * (1.0 - step / num_steps)
            elif confidence_schedule == "exponential":
                current_confidence = confidence_threshold * np.exp(-3.0 * step / num_steps)
            else:  # constant
                current_confidence = confidence_threshold
            
            generated_ids = self.generate_step(
                generated_ids, temperature, top_k, top_p, current_confidence
            )
            
            # Optional: Implement adaptive unmasking schedule
            if step < num_steps - 1:
                mask_positions = (generated_ids == self.mask_token_id)
                n_remaining = mask_positions.sum().item()
                
                if n_remaining > 0:
                    # Optionally re-mask some positions for next iteration
                    # This creates a more gradual denoising process
                    pass
        
        self.train()  # Reset to training mode
        return generated_ids
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_config_dict(self) -> dict:
        """Get complete configuration dictionary for saving/loading"""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
            'pad_token_id': self.pad_token_id,
            'mask_token_id': self.mask_token_id,
            'init_std': self.init_std,
            'rope_base': self.rope_base,
            'attention_scale': self.attention_scale,
            'qk_norm': self.qk_norm,
            'min_masks_per_sample': self.min_masks_per_sample,
            'max_masking_rate': self.max_masking_rate,
            'generation_temperature': self.generation_temperature,
            'generation_top_k': self.generation_top_k,
            'generation_top_p': self.generation_top_p,
            'transformer_config': self.transformer.get_config_dict()
        }
    
    def get_corruption_stats(self) -> dict:
        """Get corruption statistics for monitoring"""
        return self.corruption_stats.copy()
    
    def reset_corruption_stats(self):
        """Reset corruption statistics"""
        self.corruption_stats = {
            'total_masks': 0,
            'total_tokens': 0,
            'avg_masking_rate': 0.0
        }
    
    def get_memory_footprint(self) -> dict:
        """Get detailed memory footprint breakdown"""
        transformer_memory = self.transformer.get_memory_footprint()
        
        return {
            'transformer_mb': transformer_memory['total_mb'],
            'total_params': self.get_num_params(),
            'model_size_mb': self.get_num_params() * 4 / 1024**2,  # Assuming float32
            'breakdown': transformer_memory
        }
    
    def freeze_transformer_layers(self, layer_indices: List[int]):
        """Freeze specific transformer layers for experiments"""
        self.transformer.freeze_layers(layer_indices)
    
    def unfreeze_all_layers(self):
        """Unfreeze all layers"""
        self.transformer.unfreeze_all_layers()
    
    def get_layer_wise_lr_groups(self, base_lr: float, decay_factor: float = 0.9):
        """Get parameter groups for layer-wise learning rate decay"""
        return self.transformer.get_layer_wise_lr_groups(base_lr, decay_factor)
    
    def resize_token_embeddings(self, new_vocab_size: int):
        """Resize vocabulary for curriculum learning"""
        if new_vocab_size != self.vocab_size:
            self.transformer.resize_token_embeddings(new_vocab_size)
            self.vocab_size = new_vocab_size
            print(f"Resized model vocabulary to {new_vocab_size:,} tokens")
    
    def export_for_inference(self, export_path: str):
        """Export model configuration and weights for inference"""
        import json
        from pathlib import Path
        
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = self.get_config_dict()
        with open(export_path / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Save model weights
        torch.save(self.state_dict(), export_path / "model_weights.pt")
        
        # Save corruption statistics
        with open(export_path / "corruption_stats.json", "w") as f:
            json.dump(self.get_corruption_stats(), f, indent=2)
        
        print(f"Model exported to {export_path}")
    
    @classmethod
    def from_config_dict(cls, config: dict, **override_kwargs):
        """Create model from configuration dictionary"""
        # Allow overriding config values
        config = config.copy()
        config.update(override_kwargs)
        
        return cls(**config)
    
    def set_generation_defaults(self, temperature: float = 1.0, 
                               top_k: Optional[int] = None, 
                               top_p: Optional[float] = None):
        """Update default generation parameters"""
        self.generation_temperature = temperature
        self.generation_top_k = top_k
        self.generation_top_p = top_p
        print(f"Updated generation defaults: T={temperature}, top_k={top_k}, top_p={top_p}")
    
    def validate_configuration(self) -> bool:
        """Validate model configuration for consistency"""
        checks = []
        
        # Basic parameter checks
        checks.append(("vocab_size > 0", self.vocab_size > 0))
        checks.append(("d_model > 0", self.d_model > 0))
        checks.append(("n_layers > 0", self.n_layers > 0))
        checks.append(("n_heads > 0", self.n_heads > 0))
        checks.append(("d_model % n_heads == 0", self.d_model % self.n_heads == 0))
        
        # Token ID checks
        checks.append(("pad_token_id < vocab_size", self.pad_token_id < self.vocab_size))
        checks.append(("mask_token_id < vocab_size", self.mask_token_id < self.vocab_size))
        checks.append(("pad_token_id != mask_token_id", self.pad_token_id != self.mask_token_id))
        
        # Configuration range checks
        checks.append(("0 < max_masking_rate <= 1", 0 < self.max_masking_rate <= 1))
        checks.append(("min_masks_per_sample >= 1", self.min_masks_per_sample >= 1))
        checks.append(("init_std > 0", self.init_std > 0))
        checks.append(("rope_base > 0", self.rope_base > 0))
        
        # Report validation results
        all_passed = True
        for check_name, check_result in checks:
            if not check_result:
                print(f"❌ Validation failed: {check_name}")
                all_passed = False
            else:
                print(f"✅ {check_name}")
        
        return all_passed