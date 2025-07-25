#!/usr/bin/env python3
"""
Fixed Text Generation Script for Tiny Diffusion Language Model

Key fixes:
- Token ID bounds checking during generation
- Model-tokenizer vocabulary synchronization validation
- Token ID clamping to prevent UNK overflow
- Enhanced debugging output for troubleshooting
- Robust error handling for generation pipeline

Usage:
    python scripts/generate.py --checkpoint outputs/checkpoints/best_stage3.pt --prompt "The origin of"
    python scripts/generate.py --checkpoint outputs/checkpoints/best_stage3.pt --interactive
    python scripts/generate.py --checkpoint outputs/checkpoints/best_stage3.pt --batch prompts.txt
"""

import os
import sys
import json
import time
import random
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MaskedDiffusionLM
from src.evaluation import GenerationConfig, StyleMetrics, GenerationResult
from config.model import get_model_config
from config import ProjectConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FixedTextGenerator:
    """Enhanced text generator with bounds checking and debugging"""
    
    def __init__(self, model: MaskedDiffusionLM, tokenizer, device: str = 'auto'):
        self.model = model
        self.tokenizer = tokenizer
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Vocabulary validation and bounds
        self._validate_vocabulary()
        
    def _validate_vocabulary(self):
        """Validate model-tokenizer vocabulary synchronization"""
        model_vocab_size = self.model.vocab_size
        
        if hasattr(self.tokenizer, 'compressed_vocab'):
            tokenizer_vocab_size = len(self.tokenizer.compressed_vocab)
            self.is_compressed = True
        else:
            tokenizer_vocab_size = len(self.tokenizer.get_vocab())
            self.is_compressed = False
        
        logger.info(f"Model vocab size: {model_vocab_size}")
        logger.info(f"Tokenizer vocab size: {tokenizer_vocab_size}")
        
        # Validate synchronization
        if model_vocab_size != tokenizer_vocab_size:
            logger.warning(f"Vocab size mismatch! Model: {model_vocab_size}, Tokenizer: {tokenizer_vocab_size}")
            logger.warning("This may cause generation issues - using tokenizer size as bounds")
        
        # Set safe bounds
        self.vocab_size = tokenizer_vocab_size
        self.max_valid_token_id = self.vocab_size - 1
        
        logger.info(f"Using vocab bounds: [0, {self.max_valid_token_id}]")
        
        # Debug what token ID 0 maps to
        if self.is_compressed:
            token_0 = self.tokenizer.inverse_mapping.get(0, 'NOT_FOUND')
            logger.info(f"Token ID 0 maps to: '{token_0}'")
            # Show a few more mappings
            sample_tokens = {i: self.tokenizer.inverse_mapping.get(i, 'NOT_FOUND') for i in range(0, min(10, self.vocab_size))}
            logger.info(f"Sample token mappings: {sample_tokens}")
        else:
            logger.info(f"Token ID 0 maps to: '{self.tokenizer.decode([0])}'")
        
        logger.info(f"Using vocab bounds: [0, {self.max_valid_token_id}]")
    
    def _clamp_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Clamp token IDs to valid vocabulary range"""
        if token_ids.numel() == 0:
            return token_ids
            
        # Check for out-of-bounds tokens
        min_id = token_ids.min().item()
        max_id = token_ids.max().item()
        
        if min_id < 0 or max_id > self.max_valid_token_id:
            logger.warning(f"Out-of-bounds token IDs detected: [{min_id}, {max_id}]")
            logger.warning(f"Clamping to valid range: [0, {self.max_valid_token_id}]")
            
            # Count how many tokens will be clamped
            oob_count = ((token_ids < 0) | (token_ids > self.max_valid_token_id)).sum().item()
            logger.warning(f"Clamping {oob_count}/{token_ids.numel()} tokens")
        
        # Clamp to valid range
        clamped = torch.clamp(token_ids, 0, self.max_valid_token_id)
        return clamped
    
    def _fix_invalid_tokens(self, token_ids: List[int]) -> List[int]:
        """Fix invalid token IDs by remapping to valid tokens"""
        if not self.is_compressed:
            return token_ids
        
        fixed_tokens = []
        for token_id in token_ids:
            if token_id in self.tokenizer.inverse_mapping:
                fixed_tokens.append(token_id)
            else:
                # Remap invalid tokens to closest valid token
                if token_id == 0:
                    fixed_tokens.append(4)  # Map to comma
                elif token_id == 1:
                    fixed_tokens.append(5)  # Map to 'the'
                else:
                    # Find nearest valid token
                    valid_id = min(self.tokenizer.inverse_mapping.keys(), 
                                 key=lambda x: abs(x - token_id))
                    fixed_tokens.append(valid_id)
        
        return fixed_tokens
    
    def _debug_generation_step(self, token_ids: torch.Tensor, step: int):
        """Debug output for generation step"""
        if token_ids.numel() > 0:
            min_id = token_ids.min().item()
            max_id = token_ids.max().item()
            logger.debug(f"Step {step}: Token range [{min_id}, {max_id}], Shape: {token_ids.shape}")
            
            # Check for unusual patterns
            if (token_ids == 0).sum() > len(token_ids) * 0.8:  # >80% padding/UNK
                logger.warning(f"Step {step}: High padding/UNK ratio detected")
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig = None
    ) -> GenerationResult:
        """Generate text with enhanced bounds checking"""
        if config is None:
            config = GenerationConfig()
        
        # Set random seed if specified
        if config.seed is not None:
            torch.manual_seed(config.seed)
            random.seed(config.seed)
            np.random.seed(config.seed)
        
        start_time = time.time()
        
        try:
            # Tokenize prompt
            if self.is_compressed:
                prompt_tokens = self.tokenizer.encode(prompt)
            else:
                prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            
            # Validate prompt tokens
            if not prompt_tokens:
                logger.warning("Empty prompt tokens, using default")
                prompt_tokens = [0]  # Use padding/unknown token
            
            logger.debug(f"Prompt tokens: {prompt_tokens}")
            
            # Create input tensor
            input_ids = torch.tensor([prompt_tokens], device=self.device)
            
            # Clamp prompt tokens to be safe
            input_ids = self._clamp_token_ids(input_ids)
            
            logger.debug(f"Input shape: {input_ids.shape}")
            
            # Generate with the model
            with torch.no_grad():
                if hasattr(self.model, 'generate'):
                    # Use model's generate method
                    generated = self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=config.max_new_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        num_diffusion_steps=config.num_diffusion_steps,
                        do_sample=config.do_sample,
                    )
                else:
                    # Fallback: basic sampling
                    generated = self._basic_generate(input_ids, config)
            
            # Debug generated tokens
            self._debug_generation_step(generated, "final")
            
            # Clamp generated tokens to valid range
            generated_clamped = self._clamp_token_ids(generated)
            
            # Debug generated tokens
            logger.info(f"Sample generated token IDs: {generated_clamped[0][:10].tolist()}")
            
            # Extract only the new tokens (remove prompt)
            if generated_clamped.shape[1] > input_ids.shape[1]:
                new_tokens = generated_clamped[0, input_ids.shape[1]:].tolist()
            else:
                logger.warning("No new tokens generated")
                new_tokens = []
            
            # Debug decoded tokens
            logger.info(f"Sample new token IDs: {new_tokens[:10]}")
            
            # Fix invalid token IDs for compressed tokenizer
            if self.is_compressed:
                fixed_tokens = self._fix_invalid_tokens(new_tokens)
                logger.info(f"Fixed token IDs: {fixed_tokens[:10]}")
                new_tokens = fixed_tokens
                
                # Also fix the full sequence
                full_tokens_fixed = self._fix_invalid_tokens(generated_clamped[0].tolist())
                
            # Decode text
            if self.is_compressed:
                if 'full_tokens_fixed' in locals():
                    generated_text = self.tokenizer.decode(new_tokens)
                    full_text = self.tokenizer.decode(full_tokens_fixed)
                else:
                    generated_text = self.tokenizer.decode(new_tokens)
                    full_text = self.tokenizer.decode(generated_clamped[0].tolist())
            else:
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                full_text = self.tokenizer.decode(generated_clamped[0].tolist(), skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            
            # Calculate style metrics
            style_metrics = self._calculate_style_metrics(generated_text)
            
            # Log generation info
            logger.info(f"Generated {len(new_tokens)} new tokens in {generation_time:.2f}s")
            logger.debug(f"Generated text preview: {generated_text[:100]}...")
            
            return GenerationResult(
                prompt=prompt,
                generated_text=generated_text,
                full_text=full_text,
                generation_config=config,
                generation_time=generation_time,
                style_metrics=style_metrics
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Return minimal fallback result
            return GenerationResult(
                prompt=prompt,
                generated_text=f"[Generation failed: {str(e)}]",
                full_text=prompt + f" [Generation failed: {str(e)}]",
                generation_config=config,
                generation_time=time.time() - start_time,
                style_metrics=StyleMetrics(
                    avg_sentence_length=0.0,
                    vocab_richness_ttr=0.0,
                    vocab_richness_yule_k=0.0,
                    flesch_kincaid_grade=0.0,
                    gunning_fog_index=0.0,
                    avg_word_length=0.0,
                    punctuation_density=0.0,
                    function_word_ratio=0.0,
                    sentence_length_variance=0.0
                )
            )
    
    def _basic_generate(self, input_ids: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """Fallback generation method if model.generate() doesn't exist"""
        logger.warning("Using basic generation fallback")
        
        current_ids = input_ids.clone()
        
        for step in range(config.max_new_tokens):
            # Forward pass
            outputs = self.model(current_ids)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / config.temperature
            
            # Apply top-k filtering
            if config.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, config.top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_logits
            
            # Apply top-p filtering
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            if config.do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Clamp token to valid range
            next_token = torch.clamp(next_token, 0, self.max_valid_token_id)
            
            # Append to sequence
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
            
            # Check for EOS
            if hasattr(self.tokenizer, 'eos_token_id') and next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return current_ids
    
    def _calculate_style_metrics(self, text: str) -> StyleMetrics:
        """Calculate basic style metrics"""
        if not text.strip():
            return StyleMetrics(0,0,0,0,0,0,0,0,0)
        
        # Basic sentence splitting
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            sentences = [text]
        
        # Calculate metrics
        words = text.split()
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Vocabulary richness (Type-Token Ratio)
        unique_words = set(words)
        vocab_richness = len(unique_words) / len(words) if words else 0
        
        # Simple readability estimate
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        readability_grade = max(0, avg_sentence_length * 0.4 + avg_word_length * 0.6 - 2)
        
        # Function word ratio (rough estimate)
        function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
        function_word_count = sum(1 for word in words if word.lower() in function_words)
        function_word_ratio = function_word_count / len(words) if words else 0
        
        # Yule's K calculation (vocabulary diversity)
        word_counts = Counter(words)
        freq_spectrum = Counter(word_counts.values())
        yule_k = 0
        if len(words) > 0:
            try:
                yule_k = 10000 * (sum([freq * (i/len(words))**2 for i, freq in freq_spectrum.items()]) - 1/len(words))
            except:
                yule_k = 0
        
        # Additional metrics
        gunning_fog = max(0, 0.4 * (avg_sentence_length + 100 * (len([w for w in words if len(w) >= 3])/len(words) if words else 0)))
        punctuation_density = sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0
        sentence_length_variance = sum((s - avg_sentence_length)**2 for s in [len(sent.split()) for sent in sentences]) / len(sentences) if len(sentences) > 1 else 0
        
        return StyleMetrics(
            avg_sentence_length=avg_sentence_length,
            vocab_richness_ttr=vocab_richness,
            vocab_richness_yule_k=yule_k,
            flesch_kincaid_grade=readability_grade,
            gunning_fog_index=gunning_fog,
            avg_word_length=avg_word_length,
            punctuation_density=punctuation_density,
            function_word_ratio=function_word_ratio,
            sentence_length_variance=sentence_length_variance
        )


def load_model_and_tokenizer(checkpoint_path: str, config_path: Optional[str] = None) -> Tuple[MaskedDiffusionLM, Any, Dict]:
    """Load model and tokenizer with enhanced validation"""
    logger.info(f"Loading model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif 'config' in checkpoint:
        config = checkpoint['config']
        logger.info("Using config from checkpoint")
    else:
        logger.warning("No config found, using default")
        config = ProjectConfig.default().to_dict()
    
    # Load tokenizer
    tokenizer_path = "data/processed/compressed_tokenizer.json"
    if os.path.exists(tokenizer_path):
        from src.data import CompressedTokenizer
        tokenizer = CompressedTokenizer.load(tokenizer_path)
        logger.info(f"Loaded compressed tokenizer: {len(tokenizer.compressed_vocab)} tokens")
    else:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Using fallback GPT-2 tokenizer")
    
    # *** FIX: Synchronize model config with tokenizer's special token IDs ***
    # This is the critical fix. It ensures the model is created with the correct token IDs.
    model_config = config['model'].copy()
    if hasattr(tokenizer, 'compressed_vocab') and hasattr(tokenizer, 'token_mapping'):
        actual_vocab_size = len(tokenizer.compressed_vocab)
        model_config['vocab_size'] = actual_vocab_size
        model_config['eos_token_id'] = tokenizer.token_mapping.get('<|endoftext|>', 0)
        model_config['mask_token_id'] = tokenizer.token_mapping.get('[MASK]', 1)
        model_config['pad_token_id'] = tokenizer.token_mapping.get('[PAD]', 2)
        logger.info(f"Synchronized model config with tokenizer special tokens: PAD={model_config['pad_token_id']}")
    else:
        actual_vocab_size = len(tokenizer.get_vocab())
        model_config['vocab_size'] = actual_vocab_size

    logger.info(f"Setting model vocab_size to {actual_vocab_size}")
    
    # Initialize model with the corrected config
    model = MaskedDiffusionLM(model_config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info("Model loaded successfully")
    logger.info(f"Model output layer size: {model.lm_head.out_features}")
    logger.info(f"Expected vocab size: {actual_vocab_size}")
    return model, tokenizer, config


def single_generation(generator: FixedTextGenerator, prompt: str, config: GenerationConfig) -> GenerationResult:
    """Generate single text from prompt"""
    logger.info(f"Generating from prompt: '{prompt}'")
    
    result = generator.generate(prompt, config)
    
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    print(f"Generated Text ({result.generation_time:.2f}s):")
    print(f"{'-'*40}")
    print(result.generated_text)
    print(f"{'-'*40}")
    print(f"Style Metrics:")
    print(f"  Avg sentence length: {result.style_metrics.avg_sentence_length:.1f}")
    print(f"  Vocabulary richness: {result.style_metrics.vocab_richness_ttr:.3f}")
    print(f"  Readability grade: {result.style_metrics.flesch_kincaid_grade:.1f}")
    print(f"  Function word ratio: {result.style_metrics.function_word_ratio:.3f}")
    print(f"{'='*60}\n")
    
    return result


def interactive_session(generator: FixedTextGenerator):
    """Interactive text generation session"""
    print("\n" + "="*60)
    print("🤖 Interactive Text Generation (Fixed)")
    print("="*60)
    print("Commands:")
    print("  'quit' or 'exit' - End session")
    print("  'debug' - Toggle debug output")
    print("  'vocab' - Show vocabulary info")
    print("  'help' - Show this help")
    print("="*60)
    
    config = GenerationConfig(
        max_new_tokens=50,  # Start conservative
        temperature=0.6,
        top_p=0.85,
        top_k=20,
        do_sample=True
    )
    
    debug_mode = False
    
    while True:
        try:
            user_input = input("\n📝 Prompt: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Session ended!")
                break
            
            elif user_input.lower() == 'debug':
                debug_mode = not debug_mode
                level = logging.DEBUG if debug_mode else logging.INFO
                logging.getLogger().setLevel(level)
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
                continue
            
            elif user_input.lower() == 'vocab':
                print(f"Vocabulary size: {generator.vocab_size}")
                print(f"Valid token range: [0, {generator.max_valid_token_id}]")
                print(f"Compressed tokenizer: {generator.is_compressed}")
                continue
            
            elif user_input.lower() == 'help':
                print("\nCurrent settings:")
                print(f"  Max tokens: {config.max_new_tokens}")
                print(f"  Temperature: {config.temperature}")
                print(f"  Top-p: {config.top_p}")
                print(f"  Top-k: {config.top_k}")
                continue
            
            # Generate text
            result = generator.generate(user_input, config)
            
            print(f"\n✨ Generated ({result.generation_time:.2f}s):")
            print(f"{result.generated_text}")
            
        except KeyboardInterrupt:
            print("\n👋 Session ended!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def batch_generation(generator: FixedTextGenerator, prompts_file: str, config: GenerationConfig) -> List[GenerationResult]:
    """Generate texts from file of prompts"""
    logger.info(f"Running batch generation from: {prompts_file}")
    
    # Load prompts
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    if not prompts:
        logger.error("No prompts found in file")
        return []
    
    logger.info(f"Processing {len(prompts)} prompts...")
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Processing: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        
        result = generator.generate(prompt, config)
        results.append(result)
        
        print(f"  Generated: {result.generated_text[:100]}{'...' if len(result.generated_text) > 100 else ''}")
        print(f"  Time: {result.generation_time:.2f}s\n")
    
    return results


def main():
    """Main generation script entry point"""
    parser = argparse.ArgumentParser(
        description='Fixed Text Generation Script with Bounds Checking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single generation
  python scripts/generate.py --checkpoint outputs/checkpoints/best_stage3.pt --prompt "The origin of"
  
  # Interactive session
  python scripts/generate.py --checkpoint outputs/checkpoints/best_stage3.pt --interactive
  
  # Batch generation
  python scripts/generate.py --checkpoint outputs/checkpoints/best_stage3.pt --batch prompts.txt
        """
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', help='Path to config file (optional)')
    
    # Generation modes
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prompt', help='Single prompt for generation')
    group.add_argument('--interactive', action='store_true', help='Interactive generation session')
    group.add_argument('--batch', help='File with prompts (one per line)')
    
    # Generation parameters
    parser.add_argument('--max-tokens', type=int, default=100, help='Maximum new tokens')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.85, help='Nucleus sampling threshold')
    parser.add_argument('--top-k', type=int, default=20, help='Top-k sampling')
    parser.add_argument('--steps', type=int, default=20, help='Number of diffusion steps')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    # Output options
    parser.add_argument('--output', help='Save generation results to file')
    parser.add_argument('--device', default='auto', help='Device: cuda, cpu, or auto')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    try:
        # Load model and tokenizer
        model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, args.config)
        
        # Create generator
        generator = FixedTextGenerator(model, tokenizer, device=args.device)
        
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_diffusion_steps=args.steps,
            seed=args.seed,
            do_sample=True
        )
        
        print(f"🤖 Model loaded: {args.checkpoint}")
        print(f"⚙️  Generation config: {args.max_tokens} tokens, T={args.temperature}, p={args.top_p}, k={args.top_k}")
        
        # Execute based on mode
        if args.prompt:
            # Single generation
            result = single_generation(generator, args.prompt, gen_config)
            
            if args.output:
                output_data = {
                    'prompt': result.prompt,
                    'generated_text': result.generated_text,
                    'generation_time': result.generation_time,
                    'config': {
                        'max_new_tokens': gen_config.max_new_tokens,
                        'temperature': gen_config.temperature,
                        'top_p': gen_config.top_p,
                        'top_k': gen_config.top_k
                    }
                }
                
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"💾 Saved to: {args.output}")
        
        elif args.interactive:
            # Interactive session
            interactive_session(generator)
        
        elif args.batch:
            # Batch generation
            if not os.path.exists(args.batch):
                raise FileNotFoundError(f"Prompts file not found: {args.batch}")
            
            results = batch_generation(generator, args.batch, gen_config)
            
            if args.output:
                output_data = {
                    'results': [
                        {
                            'prompt': r.prompt,
                            'generated_text': r.generated_text,
                            'generation_time': r.generation_time
                        }
                        for r in results
                    ],
                    'config': {
                        'max_new_tokens': gen_config.max_new_tokens,
                        'temperature': gen_config.temperature,
                        'top_p': gen_config.top_p,
                        'top_k': gen_config.top_k
                    }
                }
                
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"💾 Saved to: {args.output}")
        
        else:
            # Default: interactive mode
            print("No mode specified, starting interactive session...")
            interactive_session(generator)
    
    except Exception as e:
        logger.error(f"Generation script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()