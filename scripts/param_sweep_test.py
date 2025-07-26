#!/usr/bin/env python3
"""
Parameter Sweep Testing for 95% Similarity Target

Runs multiple tests with randomized parameters to achieve high similarity
to target sentences from letter1.txt training data.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

# Assuming your project structure
from src.evaluation import EvaluationSuite, GenerationConfig, SimilarityMetrics
from src.model import MaskedDiffusionLM
from src.data import DataPipeline
from transformers import AutoTokenizer

@dataclass
class TestConfig:
    """Configuration for parameter sweep testing"""
    # Generation parameters (will be randomized)
    max_new_tokens: int = 50
    num_diffusion_steps: int = 20
    temperature: float = 0.6
    top_p: float = 0.85
    top_k: int = 20
    
    # Training parameters (for retraining tests)
    learning_rate: float = 1e-4
    batch_size: int = 4
    epochs: int = 10
    
    # Target similarity threshold
    similarity_threshold: float = 0.95

class ParameterSweepTester:
    """Test different parameter combinations to achieve high similarity"""
    
    def __init__(self, model_path: str, letter1_path: str = "letter1.txt"):
        self.model_path = model_path
        self.letter1_path = letter1_path
        
        # Load target sentences from letter1.txt
        self.target_sentences = self._load_target_sentences()
        
        # Initialize model and evaluation
        self.model = None
        self.tokenizer = None
        self.evaluator = None
        self._setup_model()
    
    def _load_target_sentences(self) -> List[str]:
        """Load and parse target sentences from letter1.txt"""
        try:
            with open(self.letter1_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split into sentences and filter
            import nltk
            sentences = nltk.sent_tokenize(text)
            
            # Filter for meaningful sentences (length 10-100 words)
            filtered = []
            for sent in sentences:
                words = sent.strip().split()
                if 10 <= len(words) <= 100 and sent.strip().endswith(('.', '!', '?')):
                    filtered.append(sent.strip())
            
            print(f"Loaded {len(filtered)} target sentences from {self.letter1_path}")
            return filtered
            
        except FileNotFoundError:
            print(f"Warning: {self.letter1_path} not found. Using sample sentences.")
            return [
                "I must own I felt a little proud when my captain offered me the second dignity in the vessel.",
                "My courage and my resolution is firm; but my hopes fluctuate, and my spirits are often depressed.",
                "The cold is not excessive, if you are wrapped in furs—a dress which I have already adopted."
            ]
    
    def _setup_model(self):
        """Initialize model and evaluation suite"""
        # You'll need to adapt this to your actual model loading
        print("Loading model and tokenizer...")
        # self.model = MaskedDiffusionLM.from_pretrained(self.model_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # self.evaluator = EvaluationSuite(self.model, self.tokenizer, None)
        print("Model loaded (placeholder - adapt to your setup)")
    
    def generate_random_config(self) -> TestConfig:
        """Generate randomized test configuration"""
        config = TestConfig()
        
        # Randomize generation parameters
        config.max_new_tokens = random.randint(30, 100)
        config.num_diffusion_steps = random.choice([10, 15, 20, 25, 30, 50])
        config.temperature = round(random.uniform(0.3, 1.2), 2)
        config.top_p = round(random.uniform(0.7, 0.95), 2)
        config.top_k = random.choice([10, 15, 20, 30, 40, 50])
        
        # Randomize training parameters (for retraining experiments)
        config.learning_rate = random.choice([5e-5, 1e-4, 2e-4, 5e-4])
        config.batch_size = random.choice([2, 4, 8])
        config.epochs = random.randint(5, 20)
        
        return config
    
    def test_single_configuration(self, config: TestConfig, target_sentence: str) -> Dict[str, Any]:
        """Test a single parameter configuration against target sentence"""
        
        # Extract prompt from target (first 3-5 words)
        words = target_sentence.split()
        prompt_length = random.randint(3, min(5, len(words) - 2))
        prompt = ' '.join(words[:prompt_length])
        
        print(f"Testing: prompt='{prompt}' | temp={config.temperature} | steps={config.num_diffusion_steps}")
        
        # Generate text with current config
        gen_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            num_diffusion_steps=config.num_diffusion_steps,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            seed=random.randint(1, 10000)
        )
        
        # Placeholder for actual generation - adapt to your model
        generated_text = f"{prompt} [GENERATED TEXT - IMPLEMENT WITH YOUR MODEL]"
        
        # Calculate similarity metrics
        # similarity = self.evaluator.analyzer.compare_texts(generated_text, target_sentence)
        
        # Placeholder similarity calculation
        similarity = SimilarityMetrics(
            semantic_similarity=random.uniform(0.6, 0.98),
            lexical_similarity=random.uniform(0.4, 0.9),
            syntactic_similarity=random.uniform(0.5, 0.95),
            style_similarity=random.uniform(0.7, 0.96)
        )
        
        # Calculate combined similarity score
        combined_similarity = (
            similarity.semantic_similarity * 0.4 +
            similarity.lexical_similarity * 0.2 +
            similarity.syntactic_similarity * 0.2 +
            similarity.style_similarity * 0.2
        )
        
        return {
            'config': asdict(config),
            'prompt': prompt,
            'target_sentence': target_sentence,
            'generated_text': generated_text,
            'similarity_metrics': asdict(similarity),
            'combined_similarity': combined_similarity,
            'success': combined_similarity >= config.similarity_threshold
        }
    
    def run_parameter_sweep(self, num_tests: int = 50, target_similarity: float = 0.95) -> Dict[str, Any]:
        """Run comprehensive parameter sweep"""
        
        print(f"Starting parameter sweep: {num_tests} tests targeting {target_similarity:.1%} similarity")
        
        results = []
        successful_configs = []
        best_result = None
        best_similarity = 0.0
        
        for i in range(num_tests):
            print(f"\n--- Test {i+1}/{num_tests} ---")
            
            # Generate random configuration
            config = self.generate_random_config()
            config.similarity_threshold = target_similarity
            
            # Select random target sentence
            target = random.choice(self.target_sentences)
            
            # Run test
            result = self.test_single_configuration(config, target)
            results.append(result)
            
            # Track best result
            if result['combined_similarity'] > best_similarity:
                best_similarity = result['combined_similarity']
                best_result = result
            
            # Track successful configurations
            if result['success']:
                successful_configs.append(result)
                print(f"✅ SUCCESS! Similarity: {result['combined_similarity']:.3f}")
            else:
                print(f"❌ Failed. Similarity: {result['combined_similarity']:.3f}")
        
        # Analyze results
        all_similarities = [r['combined_similarity'] for r in results]
        success_rate = len(successful_configs) / num_tests
        
        analysis = {
            'total_tests': num_tests,
            'target_similarity': target_similarity,
            'success_rate': success_rate,
            'successful_configs': len(successful_configs),
            'best_similarity': best_similarity,
            'avg_similarity': np.mean(all_similarities),
            'similarity_std': np.std(all_similarities),
            'best_result': best_result,
            'all_results': results
        }
        
        self._print_summary(analysis)
        return analysis
    
    def _print_summary(self, analysis: Dict[str, Any]):
        """Print test summary"""
        print(f"\n{'='*60}")
        print(f"PARAMETER SWEEP SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {analysis['total_tests']}")
        print(f"Target Similarity: {analysis['target_similarity']:.1%}")
        print(f"Success Rate: {analysis['success_rate']:.1%} ({analysis['successful_configs']} successes)")
        print(f"Best Similarity: {analysis['best_similarity']:.3f}")
        print(f"Average Similarity: {analysis['avg_similarity']:.3f} ± {analysis['similarity_std']:.3f}")
        
        if analysis['best_result']:
            best = analysis['best_result']
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"   Temperature: {best['config']['temperature']}")
            print(f"   Diffusion Steps: {best['config']['num_diffusion_steps']}")
            print(f"   Top-p: {best['config']['top_p']}")
            print(f"   Top-k: {best['config']['top_k']}")
            print(f"   Similarity: {best['combined_similarity']:.3f}")
    
    def save_results(self, analysis: Dict[str, Any], output_path: str = "parameter_sweep_results.json"):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Results saved to {output_path}")

def main():
    """Run parameter sweep testing"""
    
    # Configuration
    MODEL_PATH = "outputs/checkpoints/best_stage3.pt"  # Adapt to your model path
    LETTER1_PATH = "letter1.txt"
    NUM_TESTS = 100
    TARGET_SIMILARITY = 0.95
    
    # Initialize tester
    tester = ParameterSweepTester(MODEL_PATH, LETTER1_PATH)
    
    # Run parameter sweep
    results = tester.run_parameter_sweep(
        num_tests=NUM_TESTS,
        target_similarity=TARGET_SIMILARITY
    )
    
    # Save results
    tester.save_results(results)
    
    # Print insights
    print(f"\n📊 INSIGHTS:")
    if results['success_rate'] > 0:
        print(f"✅ {results['success_rate']:.1%} of configurations achieved {TARGET_SIMILARITY:.0%}+ similarity")
        print("💡 Try fine-tuning the successful parameter ranges")
    else:
        print(f"❌ No configurations achieved {TARGET_SIMILARITY:.0%} similarity")
        print("💡 Consider lowering target or adjusting parameter ranges")
        print("💡 May need more training epochs or better data preprocessing")

if __name__ == "__main__":
    main()
