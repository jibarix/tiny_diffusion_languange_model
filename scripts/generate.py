#!/usr/bin/env python3
"""
Standalone Text Generation Interface - Tiny Diffusion Project

Interactive text generation from trained models with multiple sampling strategies,
style control, and comprehensive evaluation metrics.

Usage:
    python scripts/generate.py --checkpoint best_model.pt --prompt "Science"
    python scripts/generate.py --checkpoint best_model.pt --interactive
    python scripts/generate.py --checkpoint best_model.pt --batch prompts.txt
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import ProjectConfig
from src.evaluation import TextGenerator, StyleAnalyzer, GenerationConfig, GenerationResult
from src.model import MaskedDiffusionLM
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(checkpoint_path: str, config_path: Optional[str] = None):
    """Load trained model and tokenizer from checkpoint"""
    logger.info(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load configuration
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif 'config' in checkpoint:
        config = checkpoint['config']
    else:
        logger.warning("No config found, using default")
        config = ProjectConfig.default().to_dict()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model = MaskedDiffusionLM(config['model'])
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info("Model loaded successfully")
    return model, tokenizer, config


def single_generation(generator: TextGenerator, prompt: str, config: GenerationConfig) -> GenerationResult:
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


def batch_generation(generator: TextGenerator, prompts: List[str], config: GenerationConfig) -> List[GenerationResult]:
    """Generate texts from multiple prompts"""
    logger.info(f"Running batch generation for {len(prompts)} prompts")
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Generating from: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        result = generator.generate(prompt, config)
        results.append(result)
        
        print(f"  Generated: {result.generated_text[:100]}{'...' if len(result.generated_text) > 100 else ''}")
        print(f"  Time: {result.generation_time:.2f}s")
        print()
    
    return results


def interactive_session(generator: TextGenerator):
    """Interactive text generation session"""
    print("\n" + "="*60)
    print("🤖 Interactive Text Generation")
    print("="*60)
    print("Commands:")
    print("  'quit' or 'exit' - End session")
    print("  'config' - Change generation settings")
    print("  'batch <file>' - Process file with prompts")
    print("  'save <text>' - Save last generation")
    print("  'help' - Show this help")
    print("="*60)
    
    config = GenerationConfig()
    last_result = None
    
    while True:
        try:
            user_input = input("\n📝 Prompt: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Session ended!")
                break
            
            elif user_input.lower() == 'help':
                print("\nGeneration Parameters:")
                print(f"  Max tokens: {config.max_new_tokens}")
                print(f"  Temperature: {config.temperature}")
                print(f"  Top-p: {config.top_p}")
                print(f"  Top-k: {config.top_k}")
                print(f"  Diffusion steps: {config.num_diffusion_steps}")
                continue
            
            elif user_input.lower() == 'config':
                config = configure_generation_interactive()
                print("✅ Configuration updated!")
                continue
            
            elif user_input.lower().startswith('save '):
                filename = user_input[5:].strip()
                if last_result:
                    save_generation(last_result, filename)
                else:
                    print("❌ No generation to save")
                continue
            
            elif user_input.lower().startswith('batch '):
                filename = user_input[6:].strip()
                try:
                    with open(filename, 'r') as f:
                        batch_prompts = [line.strip() for line in f if line.strip()]
                    batch_generation(generator, batch_prompts, config)
                except FileNotFoundError:
                    print(f"❌ File not found: {filename}")
                continue
            
            # Generate text
            print("🔄 Generating...")
            result = generator.generate(user_input, config)
            last_result = result
            
            # Display result
            print(f"\n📄 Generated ({result.generation_time:.2f}s):")
            print("─" * 50)
            print(result.generated_text)
            print("─" * 50)
            
            # Style info
            metrics = result.style_metrics
            print(f"📊 Style: {metrics.avg_sentence_length:.1f} avg words/sent, "
                  f"{metrics.flesch_kincaid_grade:.1f} grade level, "
                  f"{metrics.vocab_richness_ttr:.3f} vocab richness")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def configure_generation_interactive() -> GenerationConfig:
    """Interactive generation configuration"""
    print("\n🔧 Generation Configuration:")
    
    try:
        max_tokens = input("Max new tokens (100): ").strip()
        max_tokens = int(max_tokens) if max_tokens else 100
        
        temperature = input("Temperature (0.8): ").strip()
        temperature = float(temperature) if temperature else 0.8
        
        top_p = input("Top-p nucleus sampling (0.9): ").strip()
        top_p = float(top_p) if top_p else 0.9
        
        top_k = input("Top-k sampling (50): ").strip()
        top_k = int(top_k) if top_k else 50
        
        steps = input("Diffusion steps (20): ").strip()
        steps = int(steps) if steps else 20
        
        return GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_diffusion_steps=steps
        )
    except ValueError:
        print("⚠️ Invalid input, using defaults")
        return GenerationConfig()


def save_generation(result: GenerationResult, filename: str):
    """Save generation result to file"""
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'prompt': result.prompt,
        'generated_text': result.generated_text,
        'full_text': result.full_text,
        'generation_time': result.generation_time,
        'config': {
            'max_new_tokens': result.generation_config.max_new_tokens,
            'temperature': result.generation_config.temperature,
            'top_p': result.generation_config.top_p,
            'top_k': result.generation_config.top_k,
            'num_diffusion_steps': result.generation_config.num_diffusion_steps
        },
        'style_metrics': {
            'avg_sentence_length': result.style_metrics.avg_sentence_length,
            'vocab_richness_ttr': result.style_metrics.vocab_richness_ttr,
            'flesch_kincaid_grade': result.style_metrics.flesch_kincaid_grade,
            'function_word_ratio': result.style_metrics.function_word_ratio
        }
    }
    
    # Ensure .json extension
    if not filename.endswith('.json'):
        filename += '.json'
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 Saved to: {filename}")


def benchmark_generation(generator: TextGenerator, config: GenerationConfig):
    """Run generation benchmarks"""
    print("\n🏃 Running generation benchmarks...")
    
    # Test prompts covering different styles
    test_prompts = [
        "The origin of species",
        "Natural selection acts",
        "In the struggle for existence",
        "The evidence shows that",
        "We may confidently assert",
        "It is interesting to observe",
        "The facts clearly demonstrate",
        "This remarkable phenomenon"
    ]
    
    print(f"Testing with {len(test_prompts)} prompts...")
    
    # Generate texts
    results = []
    total_time = 0
    
    for prompt in test_prompts:
        result = generator.generate(prompt, config)
        results.append(result)
        total_time += result.generation_time
        print(f"  ✓ '{prompt}' -> {len(result.generated_text)} chars ({result.generation_time:.2f}s)")
    
    # Aggregate statistics
    generated_texts = [r.generated_text for r in results]
    generation_times = [r.generation_time for r in results]
    text_lengths = [len(text) for text in generated_texts]
    
    # Style analysis
    analyzer = StyleAnalyzer()
    style_metrics = [analyzer.analyze_text(text) for text in generated_texts]
    
    sentence_lengths = [m.avg_sentence_length for m in style_metrics]
    readability_scores = [m.flesch_kincaid_grade for m in style_metrics]
    vocab_richness = [m.vocab_richness_ttr for m in style_metrics]
    
    # Print results
    print(f"\n📊 Benchmark Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg time per generation: {np.mean(generation_times):.2f}s (±{np.std(generation_times):.2f})")
    print(f"  Avg text length: {np.mean(text_lengths):.0f} chars (±{np.std(text_lengths):.0f})")
    print(f"  Avg sentence length: {np.mean(sentence_lengths):.1f} words (±{np.std(sentence_lengths):.1f})")
    print(f"  Avg readability grade: {np.mean(readability_scores):.1f} (±{np.std(readability_scores):.1f})")
    print(f"  Avg vocab richness: {np.mean(vocab_richness):.3f} (±{np.std(vocab_richness):.3f})")
    
    # Sample generations
    print(f"\n📝 Sample Generations:")
    for i, result in enumerate(results[:3]):
        print(f"\n{i+1}. Prompt: '{result.prompt}'")
        print(f"   Generated: {result.generated_text[:150]}{'...' if len(result.generated_text) > 150 else ''}")


def main():
    """Main generation script entry point"""
    parser = argparse.ArgumentParser(
        description='Standalone Text Generation Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single generation
  python scripts/generate.py --checkpoint best_model.pt --prompt "The origin of"
  
  # Interactive session
  python scripts/generate.py --checkpoint best_model.pt --interactive
  
  # Batch generation from file
  python scripts/generate.py --checkpoint best_model.pt --batch prompts.txt
  
  # Custom parameters
  python scripts/generate.py --checkpoint best_model.pt --prompt "Science" --temperature 0.9 --max-tokens 200
  
  # Benchmark mode
  python scripts/generate.py --checkpoint best_model.pt --benchmark
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
    group.add_argument('--benchmark', action='store_true', help='Run generation benchmarks')
    
    # Generation parameters
    parser.add_argument('--max-tokens', type=int, default=100, help='Maximum new tokens')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling threshold')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
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
        generator = TextGenerator(model, tokenizer, device=args.device)
        
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_diffusion_steps=args.steps,
            seed=args.seed
        )
        
        print(f"🤖 Model loaded: {args.checkpoint}")
        print(f"⚙️  Generation config: {args.max_tokens} tokens, T={args.temperature}, p={args.top_p}, k={args.top_k}")
        
        # Execute based on mode
        if args.prompt:
            # Single generation
            result = single_generation(generator, args.prompt, gen_config)
            if args.output:
                save_generation(result, args.output)
        
        elif args.batch:
            # Batch generation
            with open(args.batch, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
            results = batch_generation(generator, prompts, gen_config)
            
            if args.output:
                # Save all results
                output_data = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'config': gen_config.__dict__,
                    'results': [
                        {
                            'prompt': r.prompt,
                            'generated_text': r.generated_text,
                            'generation_time': r.generation_time,
                            'style_metrics': r.style_metrics.__dict__
                        }
                        for r in results
                    ]
                }
                filename = args.output if args.output.endswith('.json') else f"{args.output}.json"
                with open(filename, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"💾 Batch results saved to: {filename}")
        
        elif args.benchmark:
            # Benchmark mode
            benchmark_generation(generator, gen_config)
        
        else:
            # Interactive mode (default)
            interactive_session(generator)
    
    except KeyboardInterrupt:
        print("\n👋 Generation interrupted!")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()