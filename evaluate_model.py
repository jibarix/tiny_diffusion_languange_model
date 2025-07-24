#!/usr/bin/env python3
"""
Standalone Evaluation Script for Trained Diffusion Model

Usage:
    python evaluate_model.py --checkpoint outputs/checkpoints/best_stage3.pt
    python evaluate_model.py --checkpoint outputs/checkpoints/best_stage3.pt --detailed
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.model import load_model_checkpoint
from src.data import CompressedTokenizer
from src.evaluation import TextGenerator, StyleAnalyzer, GenerationConfig


def load_model_and_tokenizer(checkpoint_path: str):
    """Load model and tokenizer from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load tokenizer
    tokenizer_path = "data/processed/compressed_tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    tokenizer = CompressedTokenizer.load(tokenizer_path)
    print(f"✅ Tokenizer loaded: {len(tokenizer.compressed_vocab)} tokens")
    
    # Load checkpoint manually to handle config issues
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract or infer model configuration
    if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
        config = checkpoint['config']
        if 'model' in config:
            model_config = config['model']
            print("✅ Using model config from checkpoint")
        else:
            # Config exists but model not nested
            model_config = config
            print("✅ Using flat config from checkpoint")
        
        # Ensure required fields
        model_config['vocab_size'] = len(tokenizer.compressed_vocab)
        model_config['mask_token_id'] = tokenizer.token_mapping.get('[MASK]', 1)
        model_config['pad_token_id'] = tokenizer.token_mapping.get('[PAD]', 0)
        
    else:
        print("⚠️  No config in checkpoint, inferring from state_dict...")
        
        # Infer config from state dict shapes
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Get dimensions from embedding layer
        embed_shape = state_dict['embed_tokens.weight'].shape
        vocab_size, d_model = embed_shape
        
        # Count layers
        layer_keys = [k for k in state_dict.keys() if k.startswith('layers.')]
        max_layer = max([int(k.split('.')[1]) for k in layer_keys]) + 1
        
        # Set up model config
        if d_model == 768:
            n_heads = 12
            ffn_hidden_size = 2048
        elif d_model == 512:
            n_heads = 8
            ffn_hidden_size = 1344
        else:
            n_heads = max(1, d_model // 64)
            ffn_hidden_size = d_model * 4
        
        model_config = {
            'd_model': d_model,
            'n_layers': max_layer,
            'n_heads': n_heads,
            'head_dim': d_model // n_heads,
            'ffn_hidden_size': ffn_hidden_size,
            'vocab_size': len(tokenizer.compressed_vocab),
            'max_position_embeddings': 2048,
            'attention_dropout': 0.0,
            'hidden_dropout': 0.1,
            'use_causal_mask': False,
            'mask_token_id': tokenizer.token_mapping.get('[MASK]', 1),
            'pad_token_id': tokenizer.token_mapping.get('[PAD]', 0),
            'norm_eps': 1e-6,
            'initializer_range': 0.02,
            'gradient_checkpointing': False,
            'use_cache': False,
        }
        
        print(f"✅ Inferred config: {d_model}d, {max_layer} layers, {n_heads} heads")
    
    # Create model directly
    from src.model import MaskedDiffusionLM
    model = MaskedDiffusionLM(model_config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  Architecture: {model_config['d_model']}d, {model_config['n_layers']} layers")
    
    return model, tokenizer, device


def run_generation_tests(generator: TextGenerator, prompts: List[str]) -> List[Dict]:
    """Run generation tests with multiple prompts"""
    print("\n" + "="*60)
    print("GENERATION TESTS")
    print("="*60)
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: '{prompt}'")
        print("-" * 40)
        
        try:
            config = GenerationConfig(
                max_new_tokens=80,
                temperature=0.8,
                top_p=0.9,
                do_sample=True
            )
            
            result = generator.generate(prompt, config)
            
            print(f"Generated ({result.generation_time:.2f}s):")
            print(f"'{result.generated_text}'")
            print(f"Style: {result.style_metrics.avg_sentence_length:.1f} avg words, "
                  f"{result.style_metrics.flesch_kincaid_grade:.1f} grade level")
            
            results.append({
                'prompt': prompt,
                'generated_text': result.generated_text,
                'generation_time': result.generation_time,
                'style_metrics': {
                    'avg_sentence_length': result.style_metrics.avg_sentence_length,
                    'vocab_richness': result.style_metrics.vocab_richness_ttr,
                    'readability_grade': result.style_metrics.flesch_kincaid_grade,
                    'function_word_ratio': result.style_metrics.function_word_ratio
                }
            })
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            results.append({
                'prompt': prompt,
                'error': str(e)
            })
    
    return results


def run_style_analysis(results: List[Dict]) -> Dict:
    """Analyze generated text style"""
    print("\n" + "="*60)
    print("STYLE ANALYSIS")
    print("="*60)
    
    successful_results = [r for r in results if 'generated_text' in r]
    
    if not successful_results:
        print("❌ No successful generations to analyze")
        return {}
    
    # Aggregate metrics
    avg_sentence_length = np.mean([r['style_metrics']['avg_sentence_length'] for r in successful_results])
    avg_vocab_richness = np.mean([r['style_metrics']['vocab_richness'] for r in successful_results])
    avg_readability = np.mean([r['style_metrics']['readability_grade'] for r in successful_results])
    avg_function_words = np.mean([r['style_metrics']['function_word_ratio'] for r in successful_results])
    avg_generation_time = np.mean([r['generation_time'] for r in successful_results])
    
    analysis = {
        'num_samples': len(successful_results),
        'avg_sentence_length': avg_sentence_length,
        'avg_vocab_richness': avg_vocab_richness,
        'avg_readability_grade': avg_readability,
        'avg_function_word_ratio': avg_function_words,
        'avg_generation_time': avg_generation_time
    }
    
    print(f"Samples analyzed: {analysis['num_samples']}")
    print(f"Average sentence length: {avg_sentence_length:.2f} words")
    print(f"Average vocabulary richness: {avg_vocab_richness:.3f}")
    print(f"Average readability grade: {avg_readability:.1f}")
    print(f"Average function word ratio: {avg_function_words:.3f}")
    print(f"Average generation time: {avg_generation_time:.2f}s")
    
    return analysis


def run_diversity_test(generator: TextGenerator, prompt: str, num_samples: int = 5) -> Dict:
    """Test generation diversity from same prompt"""
    print("\n" + "="*60)
    print(f"DIVERSITY TEST: '{prompt}'")
    print("="*60)
    
    generations = []
    
    for i in range(num_samples):
        try:
            config = GenerationConfig(
                max_new_tokens=60,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                seed=None  # Different seed each time
            )
            
            result = generator.generate(prompt, config)
            generations.append(result.generated_text)
            print(f"[{i+1}] {result.generated_text}")
            
        except Exception as e:
            print(f"[{i+1}] ❌ Failed: {e}")
    
    # Simple diversity measure: unique first words
    if generations:
        first_words = [gen.split()[0] if gen.split() else "" for gen in generations]
        unique_starts = len(set(first_words))
        diversity_ratio = unique_starts / len(generations)
        
        print(f"\nDiversity Analysis:")
        print(f"  Unique starting words: {unique_starts}/{len(generations)}")
        print(f"  Diversity ratio: {diversity_ratio:.2f}")
        
        return {
            'num_generations': len(generations),
            'unique_starts': unique_starts,
            'diversity_ratio': diversity_ratio,
            'generations': generations
        }
    
    return {'error': 'No successful generations'}


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate trained diffusion model')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--detailed', action='store_true', help='Run detailed analysis')
    parser.add_argument('--output', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    try:
        # Load model
        model, tokenizer, device = load_model_and_tokenizer(args.checkpoint)
        generator = TextGenerator(model, tokenizer, device)
        
        # Test prompts (Frankenstein-themed)
        prompts = [
            "The creature",
            "Natural selection",
            "The origin of",
            "In the struggle for existence",
            "The monster",
            "Victor Frankenstein",
            "The laboratory",
            "Lightning struck"
        ]
        
        # Run basic generation tests
        generation_results = run_generation_tests(generator, prompts)
        
        # Style analysis
        style_analysis = run_style_analysis(generation_results)
        
        # Diversity test if detailed
        diversity_results = {}
        if args.detailed:
            diversity_results = run_diversity_test(generator, "The creature", num_samples=5)
        
        # Compile results
        evaluation_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'checkpoint': args.checkpoint,
            'model_info': {
                'parameters': model.get_num_params(),
                'vocab_size': len(tokenizer.compressed_vocab),
                'device': str(device)
            },
            'generation_results': generation_results,
            'style_analysis': style_analysis,
            'diversity_results': diversity_results
        }
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            print(f"\n✅ Results saved to: {args.output}")
        
        # Summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        successful_gens = len([r for r in generation_results if 'generated_text' in r])
        print(f"✅ Successful generations: {successful_gens}/{len(prompts)}")
        if style_analysis:
            print(f"📝 Average text quality: {style_analysis['avg_readability_grade']:.1f} grade level")
            print(f"⚡ Average generation speed: {style_analysis['avg_generation_time']:.2f}s")
        print(f"🎯 Model ready for use!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()