"""
Text Generation Engine
Inference pipeline for masked diffusion models
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
from config import GenerationConfig


class DiffusionGenerator:
    """Text generation using masked diffusion model"""
    
    def __init__(self, model, tokenizer, device='cuda', vocab_level: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or GenerationConfig.default()  # Use config instead of hardcoded values
        self.vocab_level = vocab_level
        
        self.model.eval()
        self.model.to(device)
        
        # Token IDs
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.bos_token_id = getattr(tokenizer, 'bos_token_id', None)
        self.eos_token_id = getattr(tokenizer, 'eos_token_id', None)

        # Validate vocab size matches model
        model_vocab_size = model.vocab_size
        tokenizer_vocab_size = len(tokenizer)
        if model_vocab_size != tokenizer_vocab_size:
            print(f"Warning: Model vocab size ({model_vocab_size:,}) != "
                  f"Tokenizer vocab size ({tokenizer_vocab_size:,})")
            
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, data_dir: str, 
                       vocab_level: int = 5, device: str = 'cuda'):
        """Create generator from checkpoint with appropriate tokenizer"""
        from pathlib import Path
        import torch
        from transformers import AutoTokenizer
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        
        # Load appropriate tokenizer
        tokenizer_path = Path(data_dir) / f"tokenizer_level_{vocab_level}"
        if not tokenizer_path.exists():
            tokenizer_path = Path(data_dir) / "tokenizer_level_1"
        
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            # Fallback
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            special_tokens = {"pad_token": "<pad>", "mask_token": "<mask>", 
                             "bos_token": "<bos>", "eos_token": "<eos>"}
            tokens_to_add = {k: v for k, v in special_tokens.items() 
                            if getattr(tokenizer, k) is None}
            if tokens_to_add:
                tokenizer.add_special_tokens(tokens_to_add)
        
        # Create model
        from model.diffusion import MaskedDiffusionLM
        model = MaskedDiffusionLM(
            vocab_size=len(tokenizer),
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            d_ff=config.model.d_ff,
            max_seq_len=config.model.max_seq_len,
            dropout=config.model.dropout,
            attention_dropout=config.model.attention_dropout,
            use_bias=config.model.use_bias,
            norm_eps=config.model.norm_eps,
            pad_token_id=tokenizer.pad_token_id,
            mask_token_id=tokenizer.mask_token_id
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, tokenizer, device, vocab_level)
    
    def generate(self,
                prompt: str = "",
                max_length = kwargs.get('max_length', self.config.max_length),
                num_steps = kwargs.get('num_steps', self.config.num_steps),
                temperature = kwargs.get('temperature', self.config.temperature),
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                num_return_sequences: int = 1) -> List[str]:
        """
        Generate text using iterative demasking
        
        Args:
            prompt: Starting prompt text
            max_length: Maximum sequence length
            num_steps: Number of denoising steps
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated text strings
        """
        with torch.no_grad():
            # Tokenize prompt
            if prompt:
                prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
                prompt_length = len(prompt_tokens)
            else:
                prompt_tokens = []
                prompt_length = 0
            
            generated_sequences = []
            
            for _ in range(num_return_sequences):
                # Initialize sequence
                input_ids = self._initialize_sequence(prompt_tokens, max_length)
                
                # Iterative demasking
                for step in range(num_steps):
                    input_ids = self._denoising_step(
                        input_ids, temperature, top_k, top_p, step, num_steps
                    )
                
                # Decode to text
                generated_text = self._decode_sequence(input_ids, prompt_length)
                generated_sequences.append(generated_text)
            
            return generated_sequences
    
    def _initialize_sequence(self, prompt_tokens: List[int], max_length: int) -> torch.Tensor:
        """Initialize sequence with prompt and masks"""
        sequence = torch.full((1, max_length), self.pad_token_id, device=self.device)
        
        # Add prompt if provided
        if prompt_tokens:
            prompt_len = min(len(prompt_tokens), max_length)
            sequence[0, :prompt_len] = torch.tensor(prompt_tokens[:prompt_len], device=self.device)
            # Fill rest with masks
            sequence[0, prompt_len:] = self.mask_token_id
        else:
            # Start with all masks (except padding)
            sequence[0, :max_length] = self.mask_token_id
        
        return sequence
    
    def _denoising_step(self, input_ids: torch.Tensor, temperature: float,
                       top_k: Optional[int], top_p: Optional[float],
                       step: int, total_steps: int) -> torch.Tensor:
        """Single denoising step"""
        
        # Find masked positions
        mask_positions = (input_ids == self.mask_token_id)
        
        if not mask_positions.any():
            return input_ids
        
        # Forward pass
        attention_mask = (input_ids != self.pad_token_id).float()
        logits = self.model.transformer(input_ids, attention_mask)
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply sampling filters
        if top_k is not None:
            logits = self._apply_top_k(logits, top_k)
        
        if top_p is not None:
            logits = self._apply_top_p(logits, top_p)
        
        # Sample tokens
        probs = F.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(input_ids.shape)
        
        # Decide which masks to fill based on confidence
        confidence_schedule = self._get_confidence_schedule(step, total_steps)
        input_ids = self._selective_unmasking(input_ids, sampled_tokens, probs, confidence_schedule)
        
        return input_ids
    
    def _apply_top_k(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k filtering"""
        if k <= 0:
            return logits
        
        k = min(k, logits.size(-1))
        top_k_values, _ = torch.topk(logits, k, dim=-1)
        min_values = top_k_values[..., [-1]]
        
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    
    def _apply_top_p(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering"""
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
    
    def _get_confidence_schedule(self, step: int, total_steps: int) -> float:
        """Get confidence threshold for unmasking at this step"""
        # Start conservative (high threshold) and become more aggressive
        progress = step / max(total_steps - 1, 1)
        
        # Exponential decay: start at 0.9, end at 0.1
        return 0.9 * np.exp(-3 * progress) + 0.1
    
    def _selective_unmasking(self, input_ids: torch.Tensor, sampled_tokens: torch.Tensor,
                           probs: torch.Tensor, confidence_threshold: float) -> torch.Tensor:
        """Selectively unmask based on model confidence"""
        mask_positions = (input_ids == self.mask_token_id)
        
        if not mask_positions.any():
            return input_ids
        
        # Get confidence scores for sampled tokens
        batch_indices = torch.arange(input_ids.size(0)).unsqueeze(1).expand_as(input_ids)
        position_indices = torch.arange(input_ids.size(1)).unsqueeze(0).expand_as(input_ids)
        confidence_scores = probs[batch_indices, position_indices, sampled_tokens]
        
        # Only unmask high-confidence predictions
        high_confidence = confidence_scores > confidence_threshold
        unmask_positions = mask_positions & high_confidence
        
        # Update input_ids
        new_input_ids = input_ids.clone()
        new_input_ids[unmask_positions] = sampled_tokens[unmask_positions]
        
        return new_input_ids
    
    def _decode_sequence(self, input_ids: torch.Tensor, prompt_length: int = 0) -> str:
        """Decode tensor to text, removing special tokens"""
        sequence = input_ids[0].cpu().numpy()
        
        # Remove padding and mask tokens
        valid_tokens = []
        for token_id in sequence:
            if token_id == self.pad_token_id or token_id == self.mask_token_id:
                break
            valid_tokens.append(token_id)
        
        # Decode
        text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
        
        # Remove prompt if we want only generated part
        if prompt_length > 0:
            # This is approximate - tokenizer encoding/decoding may not be 1:1
            prompt_text = self.tokenizer.decode(valid_tokens[:prompt_length], skip_special_tokens=True)
            if text.startswith(prompt_text):
                text = text[len(prompt_text):].strip()
        
        return text
    
    def generate_with_style_control(self,
                                  prompt: str,
                                  style_tokens: List[str],
                                  max_length: int = 512,
                                  style_strength: float = 1.2,
                                  **kwargs) -> List[str]:
        """Generate with style control using token boosting"""
        
        # Get style token IDs
        style_token_ids = set()
        for token in style_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            style_token_ids.update(token_ids)
        
        # Modified generation that boosts style tokens
        with torch.no_grad():
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = self._initialize_sequence(prompt_tokens, max_length)
            
            for step in range(kwargs.get('num_steps', 50)):
                # Standard denoising step
                input_ids = self._denoising_step_with_style(
                    input_ids, style_token_ids, style_strength, step,
                    kwargs.get('num_steps', 50), kwargs.get('temperature', 1.0),
                    kwargs.get('top_k'), kwargs.get('top_p')
                )
            
            generated_text = self._decode_sequence(input_ids, len(prompt_tokens))
            return [generated_text]
    
    def _denoising_step_with_style(self, input_ids: torch.Tensor, style_token_ids: set,
                                  style_strength: float, step: int, total_steps: int,
                                  temperature: float, top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
        """Denoising step with style token boosting"""
        
        mask_positions = (input_ids == self.mask_token_id)
        if not mask_positions.any():
            return input_ids
        
        attention_mask = (input_ids != self.pad_token_id).float()
        logits = self.model.transformer(input_ids, attention_mask)
        
        # Boost style tokens
        for token_id in style_token_ids:
            if token_id < logits.size(-1):
                logits[..., token_id] *= style_strength
        
        # Apply temperature and sampling
        if temperature != 1.0:
            logits = logits / temperature
        
        if top_k is not None:
            logits = self._apply_top_k(logits, top_k)
        
        if top_p is not None:
            logits = self._apply_top_p(logits, top_p)
        
        probs = F.softmax(logits, dim=-1)
        sampled_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(input_ids.shape)
        
        confidence_schedule = self._get_confidence_schedule(step, total_steps)
        return self._selective_unmasking(input_ids, sampled_tokens, probs, confidence_schedule)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[List[str]]:
        """Generate text for multiple prompts"""
        results = []
        
        for prompt in prompts:
            generated = self.generate(prompt=prompt, **kwargs)
            results.append(generated)
        
        return results
    
    def interactive_generation(self, initial_prompt: str = "", max_length: int = 512):
        """Interactive generation with user feedback"""
        print("Interactive generation mode. Type 'quit' to exit, 'regenerate' to try again.")
        
        current_prompt = initial_prompt
        
        while True:
            if current_prompt:
                print(f"\nPrompt: {current_prompt}")
            
            try:
                generated = self.generate(
                    prompt=current_prompt,
                    max_length=max_length,
                    num_steps=50,
                    temperature=0.8
                )[0]
                
                print(f"Generated: {generated}")
                
                user_input = input("\nOptions: (c)ontinue, (r)egenerate, (q)uit, or new prompt: ").strip().lower()
                
                if user_input in ['q', 'quit']:
                    break
                elif user_input in ['r', 'regenerate']:
                    continue
                elif user_input in ['c', 'continue']:
                    current_prompt = current_prompt + " " + generated
                else:
                    # New prompt
                    current_prompt = user_input
                    
            except KeyboardInterrupt:
                print("\nGeneration interrupted.")
                break