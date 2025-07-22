#!/usr/bin/env python3
"""
Model Evaluation Script
Comprehensive evaluation of trained diffusion model
"""

import argparse
import sys
import math
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import ProjectConfig
from model.diffusion import MaskedDiffusionLM
from evaluation.generate import DiffusionGenerator
from evaluation.metrics import StyleMetrics, StructuralMetrics
from data.pipeline import TextDataPipeline


class ModelEvaluator:
    """Comprehensive model evaluation suite"""
    
    def __init__(self, model, tokenizer, generator, config):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = generator
        self.config = config
        
        # Evaluation components
        self.style_metrics = StyleMetrics(tokenizer)
        self.structural_metrics = StructuralMetrics()
        
    def calculate_perplexity(self, texts: List[str], batch_size: int = 8) -> float:
        """Calculate perplexity on given texts"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                batch_tokens = []
                for text in batch_texts:
                    tokens = self.tokenizer.encode(text, add_special_tokens=True)
                    if len(tokens) > self.config.model.max_seq_len:
                        tokens = tokens[:self.config.model.max_seq_len]
                    batch_tokens.append(tokens)
                
                # Pad to same length
                max_len = max(len(tokens) for tokens in batch_tokens)
                input_ids = torch.zeros((len(batch_tokens), max_len), dtype=torch.long)
                attention_mask = torch.zeros_like(input_ids)
                
                for j, tokens in enumerate(batch_tokens):
                    input_ids[j, :len(tokens)] = torch.tensor(tokens)
                    attention_mask[j, :len(tokens)] = 1
                
                input_ids = input_ids.to(self.model.device)
                attention_mask = attention_mask.to(self.model.device)
                
                # Forward pass with standard masking rate
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    masking_rate=0.15  # Standard BERT rate
                )
                loss = self.model.compute_loss(outputs)
                
                # Accumulate
                valid_tokens = attention_mask.sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
        
        avg_loss = total_loss / total_tokens
        return math.exp(avg_loss)
    
    def evaluate_generation_quality(self, prompts: List[str], 
                                  num_samples: int = 3) -> Dict[str, float]:
        """Evaluate quality of generated text"""
        all_generations = []
        
        for prompt in prompts:
            generations = self.generator.generate(
                prompt=prompt,
                max_length=512,
                num_steps=50,
                temperature=0.8,
                num_return_sequences=num_samples
            )
            all_generations.extend(generations)
        
        # Calculate quality metrics
        metrics = {}
        
        # Length distribution
        lengths = [len(text.split()) for text in all_generations]
        metrics['avg_length'] = np.mean(lengths)
        metrics['length_std'] = np.std(lengths)
        
        # Diversity metrics
        unique_generations = set(all_generations)
        metrics['diversity_ratio'] = len(unique_generations) / len(all_generations)
        
        # Repetition detection
        repetition_scores = []
        for text in all_generations:
            words = text.split()
            if len(words) > 1:
                unique_words = len(set(words))
                repetition_score = 1.0 - (unique_words / len(words))
                repetition_scores.append(repetition_score)
        
        if repetition_scores:
            metrics['avg_repetition'] = np.mean(repetition_scores)
        else:
            metrics['avg_repetition'] = 0.0
        
        return metrics
    
    def evaluate_style_similarity(self, generated_texts: List[str], 
                                 reference_texts: List[str]) -> Dict[str, float]:
        """Compare generated text style to reference"""
        return self.style_metrics.compare_styles(generated_texts, reference_texts)
    
    def evaluate_structural_coherence(self, texts: List[str]) -> Dict[str, float]:
        """Evaluate structural and logical coherence"""
        return self.structural_metrics.analyze_coherence(texts)
    
    def run_full_evaluation(self, test_data: List[str], 
                           reference_data: List[str] = None,
                           output_dir: str = None) -> Dict[str, Any]:
        """Run comprehensive evaluation suite"""
        results = {}
        
        print("🔍 Running comprehensive evaluation...")
        
        # 1. Perplexity evaluation
        print("📊 Calculating perplexity...")
        start_time = time.time()
        perplexity = self.calculate_perplexity(test_data)
        results['perplexity'] = perplexity
        results['perplexity_time'] = time.time() - start_time
        print(f"   Perplexity: {perplexity:.2f}")
        
        # 2. Generation quality
        print("✨ Evaluating generation quality...")
        start_time = time.time()
        
        # Use first parts of test data as prompts
        prompts = [text.split('.')[0] + '.' for text in test_data[:10] if '.' in text]
        if not prompts:
            prompts = [text[:50] + '...' for text in test_data[:10]]
        
        quality_metrics = self.evaluate_generation_quality(prompts)
        results['generation_quality'] = quality_metrics
        results['generation_time'] = time.time() - start_time
        
        print(f"   Avg length: {quality_metrics['avg_length']:.1f} words")
        print(f"   Diversity: {quality_metrics['diversity_ratio']:.3f}")
        print(f"   Repetition: {quality_metrics['avg_repetition']:.3f}")
        
        # 3. Style similarity (if reference provided)
        if reference_data:
            print("🎨 Analyzing style similarity...")
            start_time = time.time()
            
            # Generate texts for style comparison
            style_prompts = prompts[:5]
            generated_for_style = []
            for prompt in style_prompts:
                generated = self.generator.generate(
                    prompt=prompt,
                    max_length=256,
                    num_steps=50,
                    temperature=0.8,
                    num_return_sequences=1
                )[0]
                generated_for_style.append(generated)
            
            style_metrics = self.evaluate_style_similarity(
                generated_for_style, reference_data[:20]
            )
            results['style_similarity'] = style_metrics
            results['style_analysis_time'] = time.time() - start_time
            
            print(f"   Style similarity: {style_metrics.get('overall_similarity', 0):.3f}")
        
        # 4. Structural coherence
        print("🏗️  Analyzing structural coherence...")
        start_time = time.time()
        
        # Generate longer texts for coherence analysis
        coherence_prompts = prompts[:3]
        coherence_texts = []
        for prompt in coherence_prompts:
            generated = self.generator.generate(
                prompt=prompt,
                max_length=512,
                num_steps=50,
                temperature=0.7,
                num_return_sequences=1
            )[0]
            coherence_texts.append(generated)
        
        coherence_metrics = self.evaluate_structural_coherence(coherence_texts)
        results['structural_coherence'] = coherence_metrics
        results['coherence_analysis_time'] = time.time() - start_time
        
        print(f"   Coherence score: {coherence_metrics.get('overall_coherence', 0):.3f}")
        
        # 5. Model statistics
        print("📈 Collecting model statistics...")
        results['model_stats'] = {
            'parameters': self.model.get_num_params(),
            'vocab_size': len(self.tokenizer),
            'max_seq_len': self.config.model.max_seq_len,
            'model_size_mb': self.model.get_num_params() * 4 / (1024 * 1024)  # Assuming float32
        }
        
        # Save results
        if output_dir:
            self.save_evaluation_results(results, output_dir)
        
        return results
    
    def save_evaluation_results(self, results: Dict[str, Any], output_dir: str):
        """Save evaluation results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_results = {}
        for key, value in results.items():
            if isinstance(value, (dict, list, str, int, float, bool)):
                json_results[key] = value
            else:
                json_results[key] = str(value)
        
        with open(output_path / "evaluation_results.json", "w") as f:
            json.dump(json_results, f, indent=2)
        
        # Save human-readable report
        self.generate_report(results, output_path / "evaluation_report.txt")
        
        print(f"📄 Results saved to: {output_path}")
    
    def generate_report(self, results: Dict[str, Any], report_path: Path):
        """Generate human-readable evaluation report"""
        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("TINY DIFFUSION MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Model Information
            f.write("MODEL INFORMATION\n")
            f.write("-" * 20 + "\n")
            stats = results.get('model_stats', {})
            f.write(f"Parameters: {stats.get('parameters', 0):,}\n")
            f.write(f"Model Size: {stats.get('model_size_mb', 0):.1f} MB\n")
            f.write(f"Vocabulary Size: {stats.get('vocab_size', 0):,}\n")
            f.write(f"Max Sequence Length: {stats.get('max_seq_len', 0)}\n\n")
            
            # Core Metrics
            f.write("CORE METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Perplexity: {results.get('perplexity', 0):.2f}\n")
            
            # Generation Quality
            quality = results.get('generation_quality', {})
            f.write(f"Average Length: {quality.get('avg_length', 0):.1f} words\n")
            f.write(f"Diversity Ratio: {quality.get('diversity_ratio', 0):.3f}\n")
            f.write(f"Repetition Score: {quality.get('avg_repetition', 0):.3f}\n\n")
            
            # Style Analysis
            if 'style_similarity' in results:
                f.write("STYLE ANALYSIS\n")
                f.write("-" * 20 + "\n")
                style = results['style_similarity']
                for metric, value in style.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{metric.replace('_', ' ').title()}: {value:.3f}\n")
                f.write("\n")
            
            # Structural Coherence
            coherence = results.get('structural_coherence', {})
            if coherence:
                f.write("STRUCTURAL COHERENCE\n")
                f.write("-" * 20 + "\n")
                for metric, value in coherence.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{metric.replace('_', ' ').title()}: {value:.3f}\n")
                f.write("\n")
            
            # Performance Summary
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 20 + "\n")
            total_time = (results.get('perplexity_time', 0) + 
                         results.get('generation_time', 0) +
                         results.get('style_analysis_time', 0) +
                         results.get('coherence_analysis_time', 0))
            f.write(f"Total Evaluation Time: {total_time:.1f} seconds\n")
            
            # Recommendations
            f.write("\nRECOMMENDations\n")
            f.write("-" * 20 + "\n")
            
            perp = results.get('perplexity', float('inf'))
            if perp < 10:
                f.write("✅ Excellent perplexity - model has learned the data well\n")
            elif perp < 20:
                f.write("✅ Good perplexity - model shows solid performance\n")
            elif perp < 50:
                f.write("⚠️  Moderate perplexity - consider more training\n")
            else:
                f.write("❌ High perplexity - model may need significant improvements\n")
            
            diversity = quality.get('diversity_ratio', 0)
            if diversity > 0.8:
                f.write("✅ High diversity - good generation variety\n")
            elif diversity > 0.5:
                f.write("✅ Moderate diversity - reasonable variety\n")
            else:
                f.write("⚠️  Low diversity - consider adjusting sampling parameters\n")
            
            repetition = quality.get('avg_repetition', 1)
            if repetition < 0.2:
                f.write("✅ Low repetition - natural generation\n")
            elif repetition < 0.5:
                f.write("✅ Moderate repetition - acceptable quality\n")
            else:
                f.write("⚠️  High repetition - may indicate mode collapse\n")


class StyleMetrics:
    """Style analysis metrics"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def compare_styles(self, generated_texts: List[str], 
                      reference_texts: List[str]) -> Dict[str, float]:
        """Compare stylistic features between generated and reference texts"""
        gen_features = self._extract_style_features(generated_texts)
        ref_features = self._extract_style_features(reference_texts)
        
        similarities = {}
        for feature in gen_features:
            if feature in ref_features:
                # Simple absolute difference normalized
                diff = abs(gen_features[feature] - ref_features[feature])
                max_val = max(gen_features[feature], ref_features[feature], 1e-6)
                similarities[f"{feature}_similarity"] = 1.0 - (diff / max_val)
        
        # Overall similarity (average)
        similarities['overall_similarity'] = np.mean(list(similarities.values()))
        
        return similarities
    
    def _extract_style_features(self, texts: List[str]) -> Dict[str, float]:
        """Extract stylistic features from texts"""
        features = {}
        
        if not texts:
            return features
        
        # Length statistics
        word_lengths = []
        sentence_lengths = []
        
        for text in texts:
            sentences = text.split('.')
            for sent in sentences:
                words = sent.strip().split()
                if words:
                    sentence_lengths.append(len(words))
                    word_lengths.extend([len(word) for word in words])
        
        if word_lengths:
            features['avg_word_length'] = np.mean(word_lengths)
            features['word_length_std'] = np.std(word_lengths)
        
        if sentence_lengths:
            features['avg_sentence_length'] = np.mean(sentence_lengths)
            features['sentence_length_std'] = np.std(sentence_lengths)
        
        # Vocabulary richness
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        if all_words:
            unique_words = len(set(all_words))
            features['vocabulary_richness'] = unique_words / len(all_words)
        
        return features


class StructuralMetrics:
    """Structural and coherence analysis"""
    
    def analyze_coherence(self, texts: List[str]) -> Dict[str, float]:
        """Analyze structural coherence of texts"""
        metrics = {}
        
        if not texts:
            return metrics
        
        # Sentence connectivity (simple heuristic)
        connectivity_scores = []
        for text in texts:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) > 1:
                # Count transition words/phrases
                transitions = ['however', 'therefore', 'thus', 'moreover', 
                              'furthermore', 'nevertheless', 'consequently']
                transition_count = 0
                for sent in sentences[1:]:  # Skip first sentence
                    sent_lower = sent.lower()
                    if any(trans in sent_lower for trans in transitions):
                        transition_count += 1
                
                connectivity = transition_count / (len(sentences) - 1)
                connectivity_scores.append(connectivity)
        
        if connectivity_scores:
            metrics['connectivity'] = np.mean(connectivity_scores)
        
        # Repetition patterns (negative indicator)
        repetition_scores = []
        for text in texts:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) > 1:
                # Check for repeated sentence starts
                starts = [sent.split()[:3] for sent in sentences if sent.split()]
                unique_starts = len(set(tuple(start) for start in starts))
                repetition = 1.0 - (unique_starts / len(starts)) if starts else 0
                repetition_scores.append(repetition)
        
        if repetition_scores:
            metrics['structural_repetition'] = np.mean(repetition_scores)
        
        # Overall coherence (combination of metrics)
        coherence_components = []
        if 'connectivity' in metrics:
            coherence_components.append(metrics['connectivity'])
        if 'structural_repetition' in metrics:
            coherence_components.append(1.0 - metrics['structural_repetition'])
        
        if coherence_components:
            metrics['overall_coherence'] = np.mean(coherence_components)
        
        return metrics


def load_test_data(data_dir: str, max_samples: int = 100) -> Tuple[List[str], List[str]]:
    """Load test data and reference data"""
    data_path = Path(data_dir)
    
    # Try to load processed data
    try:
        import pickle
        with open(data_path / "curriculum_splits.pkl", "rb") as f:
            splits = pickle.load(f)
        
        # Use validation data if available
        all_segments = splits.get('all', [])
        if len(all_segments) > max_samples:
            # Use last 10% as test data
            test_size = min(max_samples, len(all_segments) // 10)
            test_segments = all_segments[-test_size:]
            reference_segments = all_segments[:-test_size][:max_samples]
        else:
            test_segments = all_segments[:max_samples//2]
            reference_segments = all_segments[max_samples//2:max_samples]
        
        test_texts = [seg.text for seg in test_segments]
        reference_texts = [seg.text for seg in reference_segments]
        
        return test_texts, reference_texts
        
    except Exception as e:
        print(f"Warning: Could not load processed data: {e}")
        return [], []


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained diffusion model")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--data-dir", default="data/processed", help="Data directory")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--test-samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    
    args = parser.parse_args()
    
    print(f"🔍 Starting evaluation of: {args.checkpoint}")
    
    # Load model and tokenizer
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Path(args.data_dir) / "tokenizer")
    
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
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"📱 Using device: {device}")
    print(f"🔢 Model parameters: {model.get_num_params():,}")
    
    # Create generator
    generator = DiffusionGenerator(model, tokenizer, device)
    
    # Load test data
    print("📚 Loading test data...")
    test_data, reference_data = load_test_data(args.data_dir, args.test_samples)
    
    if not test_data:
        print("❌ No test data found. Please check data directory.")
        return
    
    print(f"📊 Loaded {len(test_data)} test samples, {len(reference_data)} reference samples")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, tokenizer, generator, config)
    
    # Run evaluation
    results = evaluator.run_full_evaluation(
        test_data=test_data,
        reference_data=reference_data if reference_data else None,
        output_dir=args.output_dir
    )
    
    # Print summary
    print("\n" + "="*50)
    print("🎯 EVALUATION SUMMARY")
    print("="*50)
    print(f"Perplexity: {results.get('perplexity', 0):.2f}")
    
    quality = results.get('generation_quality', {})
    print(f"Generation Quality:")
    print(f"  - Average Length: {quality.get('avg_length', 0):.1f} words")
    print(f"  - Diversity: {quality.get('diversity_ratio', 0):.3f}")
    print(f"  - Repetition: {quality.get('avg_repetition', 0):.3f}")
    
    if 'style_similarity' in results:
        style_sim = results['style_similarity'].get('overall_similarity', 0)
        print(f"Style Similarity: {style_sim:.3f}")
    
    coherence = results.get('structural_coherence', {})
    if 'overall_coherence' in coherence:
        print(f"Structural Coherence: {coherence['overall_coherence']:.3f}")
    
    print("✅ Evaluation completed!")


if __name__ == "__main__":
    main()