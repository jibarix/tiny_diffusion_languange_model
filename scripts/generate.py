#!/usr/bin/env python3
"""
Text Generation Script
Generate text using trained diffusion model
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import ProjectConfig
from model.diffusion import MaskedDiffusionLM
from evaluation.generate import DiffusionGenerator


def load_model_and_tokenizer(checkpoint_path: str, data_dir: str):
    """Load trained model and tokenizer"""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Load tokenizer
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(str(Path(data_dir) / "tokenizer_level_1"))
    
    # Create model
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
    
    return model, tokenizer, config


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--data-dir", default="data/processed", help="Data directory")
    parser.add_argument("--prompt", default="", help="Generation prompt")
    parser.add_argument("--max-length", type=int, default=512, help="Max generation length")
    parser.add_argument("--num-steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, help="Nucleus sampling")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--output", help="Save samples to file")
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.checkpoint}")
    
    # Load model
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, args.data_dir)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create generator
    generator = DiffusionGenerator(model, tokenizer, device)
    
    if args.interactive:
        generator.interactive_generation(args.prompt, args.max_length)
    else:
        print(f"Generating with prompt: '{args.prompt}'")
        
        samples = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            num_steps=args.num_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_samples
        )
        
        print("\n" + "="*50)
        for i, sample in enumerate(samples, 1):
            print(f"Sample {i}:")
            print(sample)
            print("-" * 40)
        
        if args.output:
            with open(args.output, 'w') as f:
                for i, sample in enumerate(samples, 1):
                    f.write(f"Sample {i}:\n{sample}\n\n")
            print(f"Samples saved to: {args.output}")


if __name__ == "__main__":
    main()