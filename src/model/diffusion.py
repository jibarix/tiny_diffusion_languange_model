"""
Masked Diffusion Language Model
Orchestrates transformer for diffusion training and inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from .transformer import TransformerModel


class MaskedDiffusionLM(nn.Module):
    """Masked Diffusion Language Model"""
    
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
                 mask_token_id: int = 1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.max_seq_len = max_seq_len
        
        # Transformer backbone
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
            pad_token_id=pad_token_id
        )
        
    def create_corruption_mask(self, 
                              input_ids: torch.Tensor,
                              masking_rate: float,
                              min_masks: int = 1) -> torch.Tensor:
        """Create random corruption mask"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
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
            
            # Randomly select positions to mask
            mask_indices = torch.randperm(n_valid)[:n_mask]
            masks[i, valid_pos[mask_indices]] = True
            
        return masks
    
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
                corruption_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference
        
        Args:
            input_ids: [batch, seq_len] - clean input tokens
            attention_mask: [batch, seq_len] - attention mask
            masking_rate: float - random masking rate (for training)
            corruption_mask: [batch, seq_len] - explicit mask (overrides masking_rate)
        """
        
        # Training mode: apply random corruption
        if self.training and masking_rate is not None:
            corruption_mask = self.create_corruption_mask(input_ids, masking_rate)
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
        
        outputs = {
            'logits': logits,
            'targets': targets,
            'corruption_mask': corruption_mask,
            'corrupted_input': corrupted_input
        }
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute diffusion training loss"""
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
        
        # Compute cross-entropy only on masked positions
        loss = F.cross_entropy(
            flat_logits[loss_mask],
            flat_targets[loss_mask],
            reduction='mean'
        )
        
        return loss
    
    def generate_step(self,
                     input_ids: torch.Tensor,
                     temperature: float = 1.0,
                     top_k: Optional[int] = None,
                     top_p: Optional[float] = None) -> torch.Tensor:
        """Single denoising step during generation"""
        self.eval()
        
        with torch.no_grad():
            # Find masked positions
            mask_positions = (input_ids == self.mask_token_id)
            
            if not mask_positions.any():
                return input_ids  # No masks to fill
            
            # Forward pass
            outputs = self.forward(input_ids)
            logits = outputs['logits']  # [batch, seq_len, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., [-1]]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            sampled_tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(input_ids.shape)
            
            # Only update masked positions
            new_input_ids = input_ids.clone()
            new_input_ids[mask_positions] = sampled_tokens[mask_positions]
            
            return new_input_ids
    
    def generate(self,
                prompt_ids: torch.Tensor,
                max_length: int,
                num_steps: int = 50,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None) -> torch.Tensor:
        """Generate text using iterative demasking"""
        self.eval()
        
        batch_size = prompt_ids.size(0)
        device = prompt_ids.device
        
        # Start with fully masked sequence (except prompt)
        generated_ids = torch.full((batch_size, max_length), self.mask_token_id, device=device)
        generated_ids[:, :prompt_ids.size(1)] = prompt_ids
        
        # Iteratively demask
        for step in range(num_steps):
            generated_ids = self.generate_step(
                generated_ids, temperature, top_k, top_p
            )
            
            # Optionally adjust unmasking schedule
            if step < num_steps - 1:
                # Keep some positions masked for next iteration
                mask_positions = (generated_ids == self.mask_token_id)
                n_remaining = mask_positions.sum().item()
                
                if n_remaining > 0:
                    # Unmask more tokens as we progress
                    unmask_ratio = 1.0 - (step + 1) / num_steps
                    n_keep_masked = int(n_remaining * unmask_ratio)
                    
                    if n_keep_masked > 0:
                        # Randomly re-mask some positions
                        for i in range(batch_size):
                            row_masks = mask_positions[i].nonzero(as_tuple=True)[0]
                            if len(row_masks) > n_keep_masked:
                                keep_indices = torch.randperm(len(row_masks))[:n_keep_masked]
                                unmask_indices = row_masks[~torch.isin(torch.arange(len(row_masks)), keep_indices)]
                                # These remain unmasked, others get remasked
        
        return generated_ids
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())