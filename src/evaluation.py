"""
Generation + Style Analysis + Benchmarking

Complete evaluation suite for tiny masked diffusion language model:
- Text generation with multiple sampling strategies
- Stylometric analysis and similarity metrics
- Performance benchmarking and comparison tools
- Interactive generation and analysis tools

Based on 2025 research on diffusion model evaluation and style transfer.
"""

import os
import re
import json
import math
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import Counter
import numpy as np
import pandas as pd

# Core dependencies
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# NLP analysis
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import textstat

# Local imports
from .model import MaskedDiffusionLM
from .data import DataPipeline, TextSegment

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 100
    num_diffusion_steps: int = 20
    temperature: float = 0.6
    top_p: float = 0.85
    top_k: int = 20
    do_sample: bool = True
    num_return_sequences: int = 1
    seed: Optional[int] = None


@dataclass
class StyleMetrics:
    """Stylometric analysis results"""
    avg_sentence_length: float
    vocab_richness_ttr: float  # Type-Token Ratio
    vocab_richness_yule_k: float  # Yule's K
    flesch_kincaid_grade: float
    gunning_fog_index: float
    avg_word_length: float
    punctuation_density: float
    function_word_ratio: float
    sentence_length_variance: float


@dataclass
class SimilarityMetrics:
    """Similarity comparison results"""
    semantic_similarity: float  # Sentence embedding cosine similarity
    lexical_similarity: float   # Jaccard similarity
    syntactic_similarity: float # POS tag similarity
    style_similarity: float     # Combined stylometric similarity


@dataclass
class GenerationResult:
    """Complete generation result with metadata"""
    prompt: str
    generated_text: str
    full_text: str
    generation_config: GenerationConfig
    generation_time: float
    style_metrics: StyleMetrics
    prompt_similarity: Optional[SimilarityMetrics] = None


class TextGenerator:
    """Inference pipeline with multiple sampling strategies"""
    
    def __init__(self, model: MaskedDiffusionLM, tokenizer, device: str = 'auto'):
        self.model = model
        self.tokenizer = tokenizer
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig = None
    ) -> GenerationResult:
        """Generate text from prompt using masked diffusion"""
        if config is None:
            config = GenerationConfig()
        
        # Set random seed if specified
        if config.seed is not None:
            torch.manual_seed(config.seed)
            random.seed(config.seed)
            np.random.seed(config.seed)
        
        import time
        start_time = time.time()
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            # Generate using diffusion
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=config.max_new_tokens,
                num_diffusion_steps=config.num_diffusion_steps,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.do_sample,
            )
        
        # --- FIX: Correctly extract the newly generated part of the text ---
        # The old method used string slicing which is unreliable.
        # This new method works at the token level for perfect extraction.
        prompt_token_count = input_ids.shape[1]
        new_token_ids = generated_ids[0, prompt_token_count:].tolist()
        generated_text = self.tokenizer.decode(new_token_ids).strip()
        full_text = self.tokenizer.decode(generated_ids[0].tolist())
        # --- END FIX ---
        
        generation_time = time.time() - start_time
        
        # Analyze generated text
        analyzer = StyleAnalyzer()
        style_metrics = analyzer.analyze_text(generated_text)
        
        return GenerationResult(
            prompt=prompt,
            generated_text=generated_text,
            full_text=full_text,
            generation_config=config,
            generation_time=generation_time,
            style_metrics=style_metrics
        )
    
    def generate_multiple(
        self,
        prompt: str,
        num_samples: int = 5,
        config: GenerationConfig = None
    ) -> List[GenerationResult]:
        """Generate multiple samples from the same prompt"""
        results = []
        
        for i in range(num_samples):
            # Use different seeds for variety
            if config and config.seed is not None:
                sample_config = GenerationConfig(**asdict(config))
                sample_config.seed = config.seed + i
            else:
                sample_config = config
            
            result = self.generate(prompt, sample_config)
            results.append(result)
        
        return results
    
    def interactive_generation(self):
        """Interactive text generation session"""
        print("=== Interactive Text Generation ===")
        print("Type 'quit' to exit, 'config' to change settings")
        
        config = GenerationConfig()
        
        while True:
            try:
                user_input = input("\nPrompt: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'config':
                    config = self._configure_generation()
                    continue
                elif not user_input:
                    continue
                
                print("Generating...")
                result = self.generate(user_input, config)
                
                print(f"\nGenerated ({result.generation_time:.2f}s):")
                print("-" * 50)
                print(result.generated_text)
                print("-" * 50)
                print(f"Style: {result.style_metrics.avg_sentence_length:.1f} avg sent len, "
                      f"{result.style_metrics.flesch_kincaid_grade:.1f} grade level")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Session ended.")
    
    def _configure_generation(self) -> GenerationConfig:
        """Interactive configuration of generation parameters"""
        print("\nGeneration Configuration:")
        
        try:
            max_tokens = int(input(f"Max new tokens (default 100): ") or "100")
            steps = int(input(f"Diffusion steps (default 20): ") or "20")
            temperature = float(input(f"Temperature (default 0.8): ") or "0.8")
            top_p = float(input(f"Top-p (default 0.9): ") or "0.9")
            top_k = int(input(f"Top-k (default 50): ") or "50")
            
            return GenerationConfig(
                max_new_tokens=max_tokens,
                num_diffusion_steps=steps,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        except ValueError:
            print("Invalid input, using defaults")
            return GenerationConfig()


class StyleAnalyzer:
    """Stylometric analysis and similarity metrics"""
    
    def __init__(self):
        # Initialize NLP tools
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("spaCy model 'en_core_web_sm' not found")
        
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Function words for analysis
        self.function_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would',
            'there', 'their'
        }
    
    def analyze_text(self, text: str) -> StyleMetrics:
        """Comprehensive stylometric analysis"""
        if not text or not text.strip():
            return StyleMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Basic preprocessing
        doc = self.nlp(text)
        sentences = list(doc.sents)
        words = [token.text.lower() for token in doc if token.is_alpha]
        
        if not words or not sentences:
            return StyleMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Sentence-level metrics
        sentence_lengths = [len([t for t in sent if t.is_alpha]) for sent in sentences]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        sentence_length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Vocabulary richness
        vocab_size = len(set(words))
        total_words = len(words)
        ttr = vocab_size / total_words if total_words > 0 else 0
        
        # Yule's K (vocabulary diversity)
        word_counts = Counter(words)
        freq_spectrum = Counter(word_counts.values())
        yule_k = 0
        if total_words > 0:
            try:
                yule_k = 10000 * (sum([freq * (i/total_words)**2 for i, freq in freq_spectrum.items()]) - 1/total_words)
            except:
                yule_k = 0
        
        # Readability
        try:
            flesch_kincaid = textstat.flesch_kincaid_grade(text)
            gunning_fog = textstat.gunning_fog(text)
        except:
            flesch_kincaid = 0
            gunning_fog = 0
        
        # Word length
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Punctuation density
        punctuation_marks = sum(1 for token in doc if token.is_punct)
        punctuation_density = punctuation_marks / len(doc) if len(doc) > 0 else 0
        
        # Function word ratio
        function_word_count = sum(1 for word in words if word in self.function_words)
        function_word_ratio = function_word_count / total_words if total_words > 0 else 0
        
        return StyleMetrics(
            avg_sentence_length=avg_sentence_length,
            vocab_richness_ttr=ttr,
            vocab_richness_yule_k=yule_k,
            flesch_kincaid_grade=flesch_kincaid,
            gunning_fog_index=gunning_fog,
            avg_word_length=avg_word_length,
            punctuation_density=punctuation_density,
            function_word_ratio=function_word_ratio,
            sentence_length_variance=sentence_length_variance
        )
    
    def compare_texts(self, text1: str, text2: str) -> SimilarityMetrics:
        """Compare stylistic similarity between two texts"""
        # Semantic similarity using embeddings
        embeddings1 = self.embedding_model.encode([text1])
        embeddings2 = self.embedding_model.encode([text2])
        semantic_sim = cosine_similarity(embeddings1, embeddings2)[0][0]
        
        # Lexical similarity (Jaccard index)
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        
        words1 = set(token.lemma_.lower() for token in doc1 if token.is_alpha)
        words2 = set(token.lemma_.lower() for token in doc2 if token.is_alpha)
        
        if not words1 and not words2:
            lexical_sim = 1.0
        elif not words1 or not words2:
            lexical_sim = 0.0
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            lexical_sim = intersection / union if union > 0 else 0
        
        # Syntactic similarity (POS tag distribution)
        pos1 = [token.pos_ for token in doc1 if token.is_alpha]
        pos2 = [token.pos_ for token in doc2 if token.is_alpha]
        
        pos_counts1 = Counter(pos1)
        pos_counts2 = Counter(pos2)
        
        all_pos = set(pos_counts1.keys()).union(set(pos_counts2.keys()))
        if all_pos:
            pos_vector1 = np.array([pos_counts1.get(pos, 0) for pos in all_pos])
            pos_vector2 = np.array([pos_counts2.get(pos, 0) for pos in all_pos])
            
            # Normalize
            if pos_vector1.sum() > 0:
                pos_vector1 = pos_vector1 / pos_vector1.sum()
            if pos_vector2.sum() > 0:
                pos_vector2 = pos_vector2 / pos_vector2.sum()
            
            # Cosine similarity
            if np.linalg.norm(pos_vector1) > 0 and np.linalg.norm(pos_vector2) > 0:
                syntactic_sim = np.dot(pos_vector1, pos_vector2) / (np.linalg.norm(pos_vector1) * np.linalg.norm(pos_vector2))
            else:
                syntactic_sim = 0.0
        else:
            syntactic_sim = 0.0
        
        # Style similarity (compare stylometric features)
        style1 = self.analyze_text(text1)
        style2 = self.analyze_text(text2)
        
        # Create feature vectors
        features1 = np.array([
            style1.avg_sentence_length,
            style1.vocab_richness_ttr,
            style1.flesch_kincaid_grade,
            style1.avg_word_length,
            style1.punctuation_density,
            style1.function_word_ratio
        ])
        
        features2 = np.array([
            style2.avg_sentence_length,
            style2.vocab_richness_ttr,
            style2.flesch_kincaid_grade,
            style2.avg_word_length,
            style2.punctuation_density,
            style2.function_word_ratio
        ])
        
        # Normalize features to [0, 1] range for comparison
        combined = np.vstack([features1, features2])
        if np.max(combined) > np.min(combined):
            combined_norm = (combined - np.min(combined)) / (np.max(combined) - np.min(combined))
            features1_norm = combined_norm[0]
            features2_norm = combined_norm[1]
            
            # Calculate similarity (1 - euclidean distance normalized)
            distance = np.linalg.norm(features1_norm - features2_norm)
            max_distance = np.sqrt(len(features1_norm))  # Maximum possible distance
            style_sim = max(0, 1 - distance / max_distance)
        else:
            style_sim = 1.0
        
        return SimilarityMetrics(
            semantic_similarity=float(semantic_sim),
            lexical_similarity=lexical_sim,
            syntactic_similarity=float(syntactic_sim),
            style_similarity=style_sim
        )


class Benchmarker:
    """Performance comparison and benchmarking tools"""
    
    def __init__(self, model: MaskedDiffusionLM, tokenizer, device: str = 'auto'):
        self.generator = TextGenerator(model, tokenizer, device)
        self.analyzer = StyleAnalyzer()
    
    def perplexity_benchmark(self, test_texts: List[str]) -> Dict[str, float]:
        """Calculate perplexity on test texts"""
        self.generator.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in test_texts:
                # Tokenize
                token_ids = self.generator.tokenizer.encode(text)
                if len(token_ids) < 2:
                    continue
                
                # Create input (all masked) and labels (original)
                input_ids = torch.tensor([[self.generator.tokenizer.token_mapping.get('[MASK]', 1)] * len(token_ids)], 
                                       device=self.generator.device)
                labels = torch.tensor([token_ids], device=self.generator.device)
                
                # Forward pass
                outputs = self.generator.model(input_ids=input_ids, labels=labels)
                loss = outputs['loss']
                
                total_loss += loss.item() * len(token_ids)
                total_tokens += len(token_ids)
        
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(avg_loss)
        
        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens,
            'num_texts': len(test_texts)
        }
    
    def style_fidelity_benchmark(self, reference_text: str, num_samples: int = 10) -> Dict[str, Any]:
        """Benchmark style fidelity against reference text"""
        # Extract sample prompts from reference
        sentences = nltk.sent_tokenize(reference_text)
        prompts = []
        
        for i in range(0, min(len(sentences), num_samples), max(1, len(sentences) // num_samples)):
            sentence = sentences[i]
            words = sentence.split()
            if len(words) > 5:
                # Use first 3-5 words as prompt
                prompt_length = random.randint(3, min(5, len(words) - 1))
                prompt = ' '.join(words[:prompt_length])
                prompts.append(prompt)
        
        if not prompts:
            return {'error': 'Could not extract prompts from reference text'}
        
        # Generate samples
        results = []
        similarities = []
        
        for prompt in prompts:
            result = self.generator.generate(prompt, GenerationConfig(max_new_tokens=50))
            results.append(result)
            
            # Compare with reference
            similarity = self.analyzer.compare_texts(result.generated_text, reference_text)
            similarities.append(similarity)
        
        # Aggregate metrics
        if similarities:
            avg_semantic = np.mean([s.semantic_similarity for s in similarities])
            avg_lexical = np.mean([s.lexical_similarity for s in similarities])
            avg_syntactic = np.mean([s.syntactic_similarity for s in similarities])
            avg_style = np.mean([s.style_similarity for s in similarities])
        else:
            avg_semantic = avg_lexical = avg_syntactic = avg_style = 0.0
        
        # Style consistency across generations
        generated_texts = [r.generated_text for r in results]
        style_metrics = [r.style_metrics for r in results]
        
        if len(style_metrics) > 1:
            sentence_lengths = [s.avg_sentence_length for s in style_metrics]
            vocab_richness = [s.vocab_richness_ttr for s in style_metrics]
            readability = [s.flesch_kincaid_grade for s in style_metrics]
            
            style_consistency = {
                'sentence_length_std': np.std(sentence_lengths),
                'vocab_richness_std': np.std(vocab_richness),
                'readability_std': np.std(readability)
            }
        else:
            style_consistency = {'sentence_length_std': 0, 'vocab_richness_std': 0, 'readability_std': 0}
        
        return {
            'num_samples': len(results),
            'avg_semantic_similarity': avg_semantic,
            'avg_lexical_similarity': avg_lexical,
            'avg_syntactic_similarity': avg_syntactic,
            'avg_style_similarity': avg_style,
            'style_consistency': style_consistency,
            'sample_results': results[:3]  # Include first 3 for inspection
        }
    
    def generation_diversity_benchmark(self, prompts: List[str], samples_per_prompt: int = 5) -> Dict[str, Any]:
        """Measure diversity of generations from same prompts"""
        all_diversities = []
        
        for prompt in prompts:
            # Generate multiple samples
            results = self.generator.generate_multiple(prompt, samples_per_prompt)
            generated_texts = [r.generated_text for r in results]
            
            if len(generated_texts) < 2:
                continue
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(generated_texts)):
                for j in range(i + 1, len(generated_texts)):
                    sim = self.analyzer.compare_texts(generated_texts[i], generated_texts[j])
                    similarities.append(sim.semantic_similarity)
            
            # Diversity = 1 - average similarity
            avg_similarity = np.mean(similarities) if similarities else 0
            diversity = 1 - avg_similarity
            all_diversities.append(diversity)
        
        return {
            'num_prompts': len(prompts),
            'samples_per_prompt': samples_per_prompt,
            'avg_diversity': np.mean(all_diversities) if all_diversities else 0,
            'diversity_std': np.std(all_diversities) if all_diversities else 0,
            'diversity_scores': all_diversities
        }


class EvaluationSuite:
    """Complete evaluation suite combining all analysis tools"""
    
    def __init__(self, model: MaskedDiffusionLM, tokenizer, data_pipeline: DataPipeline, device: str = 'auto'):
        self.model = model
        self.tokenizer = tokenizer
        self.data_pipeline = data_pipeline
        self.device = device
        
        self.generator = TextGenerator(model, tokenizer, device)
        self.analyzer = StyleAnalyzer()
        self.benchmarker = Benchmarker(model, tokenizer, device)
    
    def full_evaluation(self, reference_text: str = None) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        print("Running full evaluation suite...")
        
        results = {}
        
        # 1. Basic generation test
        print("1. Testing basic generation...")
        # --- MODIFICATION: Use thematic prompts from the report ---
        test_prompts = [
            "Victor felt",
            "The monster",
            "In my journal I wrote",
            "The laboratory held",
            "I created death"
        ]
        
        # --- MODIFICATION: Use optimal generation parameters from the report ---
        optimal_gen_config = GenerationConfig(
            max_new_tokens=80,          # Optimal range 60-100
            num_diffusion_steps=50,     # Optimal is 50+
            temperature=0.6,            # Optimal range 0.6-0.7
            top_k=25,                   # Optimal range 20-25
            top_p=0.85,                 # Keep as is, good value
            do_sample=True
        )
        
        generation_results = []
        for prompt in test_prompts:
            result = self.generator.generate(prompt, optimal_gen_config)
            generation_results.append(result)
        
        results['generation_test'] = {
            'prompts': test_prompts,
            'results': generation_results,
            'avg_generation_time': np.mean([r.generation_time for r in generation_results])
        }
        
        # 2. Perplexity on validation data
        print("2. Calculating perplexity...")
        if hasattr(self.data_pipeline, 'segments') and self.data_pipeline.segments:
            test_texts = [seg.text for seg in self.data_pipeline.segments[-50:]]  # Last 50 segments
            perplexity_results = self.benchmarker.perplexity_benchmark(test_texts)
            results['perplexity'] = perplexity_results
        
        # 3. Style fidelity (if reference provided)
        if reference_text:
            print("3. Analyzing style fidelity...")
            style_results = self.benchmarker.style_fidelity_benchmark(reference_text, num_samples=5)
            results['style_fidelity'] = style_results
        
        # 4. Generation diversity
        print("4. Measuring generation diversity...")
        diversity_results = self.benchmarker.generation_diversity_benchmark(test_prompts[:2], samples_per_prompt=3)
        results['diversity'] = diversity_results
        
        # 5. Style analysis of generations
        print("5. Analyzing generated text style...")
        all_generated = ' '.join([r.generated_text for r in generation_results])
        if reference_text:
            overall_similarity = self.analyzer.compare_texts(all_generated, reference_text)
            results['overall_similarity'] = asdict(overall_similarity)
        
        overall_style = self.analyzer.analyze_text(all_generated)
        results['generated_style'] = asdict(overall_style)
        
        if reference_text:
            reference_style = self.analyzer.analyze_text(reference_text)
            results['reference_style'] = asdict(reference_style)
        
        print("Evaluation complete!")
        return results
    
    def save_evaluation_report(self, results: Dict[str, Any], filepath: str):
        """Save evaluation results to file"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Evaluation report saved to: {filepath}")
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print human-readable evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        if 'generation_test' in results:
            gen_results = results['generation_test']
            print(f"Generation Test:")
            print(f"  Prompts tested: {len(gen_results['prompts'])}")
            print(f"  Avg generation time: {gen_results['avg_generation_time']:.2f}s")
        
        if 'perplexity' in results:
            ppl = results['perplexity']
            print(f"Perplexity: {ppl['perplexity']:.2f}")
        
        if 'style_fidelity' in results:
            style = results['style_fidelity']
            print(f"Style Fidelity:")
            print(f"  Semantic similarity: {style['avg_semantic_similarity']:.3f}")
            print(f"  Style similarity: {style['avg_style_similarity']:.3f}")
        
        if 'diversity' in results:
            div = results['diversity']
            print(f"Generation Diversity: {div['avg_diversity']:.3f}")
        
        if 'generated_style' in results:
            style = results['generated_style']
            print(f"Generated Text Style:")
            print(f"  Avg sentence length: {style['avg_sentence_length']:.1f}")
            print(f"  Grade level: {style['flesch_kincaid_grade']:.1f}")
            print(f"  Vocabulary richness: {style['vocab_richness_ttr']:.3f}")
        
        print("="*60)


# Example usage and testing
if __name__ == "__main__":
    print("Testing evaluation components...")
    
    # Test style analyzer
    analyzer = StyleAnalyzer()
    
    test_text1 = "The origin of species is a complex process. Natural selection drives evolution through many generations."
    test_text2 = "Species originate through complicated mechanisms. Evolution occurs via natural selection over time."
    
    style1 = analyzer.analyze_text(test_text1)
    print(f"Style analysis 1: {style1.avg_sentence_length:.1f} avg words, grade {style1.flesch_kincaid_grade:.1f}")
    
    similarity = analyzer.compare_texts(test_text1, test_text2)
    print(f"Similarity: semantic={similarity.semantic_similarity:.3f}, style={similarity.style_similarity:.3f}")
    
    print("Evaluation components test complete!")
