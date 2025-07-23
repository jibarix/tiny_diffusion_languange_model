"""
Evaluation Configuration
Parameters for model evaluation and benchmarking
"""

from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    
    # ================================
    # CORE EVALUATION PARAMETERS
    # ================================
    batch_size: int = 8                  # Evaluation batch size
    max_samples: int = 100               # Maximum test samples to evaluate
    
    # ================================
    # PERPLEXITY CALCULATION
    # ================================
    perplexity_batch_size: int = 8       # Batch size for perplexity calculation
    perplexity_masking_rate: float = 0.15  # Standard BERT masking rate for perplexity
    
    # ================================
    # GENERATION QUALITY EVALUATION
    # ================================
    generation_num_samples: int = 3      # Number of samples per prompt
    generation_max_length: int = 512     # Max generation length
    generation_num_steps: int = 50       # Diffusion steps for generation
    generation_temperature: float = 0.8  # Generation temperature
    generation_top_k: Optional[int] = None  # Top-k sampling
    generation_top_p: Optional[float] = None  # Nucleus sampling
    
    # Number of prompts to use for different evaluations
    quality_eval_prompts: int = 10       # Prompts for generation quality
    style_eval_prompts: int = 5          # Prompts for style evaluation
    coherence_eval_prompts: int = 3      # Prompts for coherence evaluation
    
    # ================================
    # STYLE ANALYSIS PARAMETERS
    # ================================
    style_reference_samples: int = 20    # Reference texts for style comparison
    style_similarity_threshold: float = 0.7  # Minimum style similarity
    
    # ================================
    # COHERENCE ANALYSIS PARAMETERS
    # ================================
    coherence_max_length: int = 512      # Max length for coherence analysis
    coherence_temperature: float = 0.7   # Temperature for coherence generation
    coherence_min_sentences: int = 3     # Minimum sentences for coherence analysis
    
    # ================================
    # DIVERSITY METRICS
    # ================================
    diversity_num_samples: int = 10      # Samples for diversity calculation
    diversity_ngram_size: int = 3        # N-gram size for diversity metrics
    repetition_penalty_threshold: float = 0.5  # Threshold for repetition detection
    
    # ================================
    # PERFORMANCE BENCHMARKING
    # ================================
    benchmark_iterations: int = 5        # Iterations for performance benchmarking
    benchmark_warmup_steps: int = 2      # Warmup steps before timing
    
    # ================================
    # OUTPUT AND REPORTING
    # ================================
    save_generated_samples: bool = True  # Save generated text samples
    save_attention_weights: bool = False # Save attention weight visualizations
    generate_plots: bool = True          # Generate evaluation plots
    detailed_metrics: bool = True        # Include detailed metric breakdowns
    
    # ================================
    # QUALITY THRESHOLDS
    # ================================
    # Thresholds for determining model quality
    excellent_perplexity: float = 10.0   # Perplexity threshold for "excellent"
    good_perplexity: float = 20.0        # Perplexity threshold for "good"
    acceptable_perplexity: float = 50.0  # Perplexity threshold for "acceptable"
    
    min_diversity_ratio: float = 0.8     # Minimum diversity ratio for "high diversity"
    moderate_diversity_ratio: float = 0.5  # Threshold for "moderate diversity"
    
    max_repetition_score: float = 0.2    # Maximum repetition for "low repetition"
    moderate_repetition_score: float = 0.5  # Threshold for "moderate repetition"
    
    # ================================
    # MEMORY AND EFFICIENCY
    # ================================
    use_cache: bool = True               # Cache evaluation results
    parallel_evaluation: bool = True     # Use parallel processing when possible
    memory_efficient_mode: bool = False  # Reduce memory usage at cost of speed
    
    # ================================
    # VOCABULARY CURRICULUM EVALUATION
    # ================================
    evaluate_all_vocab_levels: bool = True  # Evaluate all vocabulary levels
    vocab_level_comparison: bool = True   # Compare performance across vocab levels
    
    @classmethod
    def default(cls) -> 'EvaluationConfig':
        """Default evaluation configuration"""
        return cls()
    
    @classmethod
    def fast(cls) -> 'EvaluationConfig':
        """Fast evaluation for development"""
        return cls(
            batch_size=4,
            max_samples=20,
            generation_num_samples=1,
            generation_num_steps=20,
            quality_eval_prompts=3,
            style_eval_prompts=2,
            coherence_eval_prompts=1,
            benchmark_iterations=2,
            save_attention_weights=False,
            generate_plots=False,
            detailed_metrics=False
        )
    
    @classmethod
    def comprehensive(cls) -> 'EvaluationConfig':
        """Comprehensive evaluation for research"""
        return cls(
            batch_size=16,
            max_samples=500,
            generation_num_samples=5,
            generation_num_steps=100,
            quality_eval_prompts=20,
            style_eval_prompts=10,
            coherence_eval_prompts=5,
            style_reference_samples=50,
            diversity_num_samples=20,
            benchmark_iterations=10,
            save_attention_weights=True,
            generate_plots=True,
            detailed_metrics=True,
            evaluate_all_vocab_levels=True
        )
    
    @classmethod
    def memory_efficient(cls) -> 'EvaluationConfig':
        """Memory-efficient evaluation for limited hardware"""
        return cls(
            batch_size=2,
            perplexity_batch_size=2,
            max_samples=50,
            generation_num_samples=2,
            generation_max_length=256,
            quality_eval_prompts=5,
            style_eval_prompts=3,
            coherence_eval_prompts=2,
            memory_efficient_mode=True,
            parallel_evaluation=False,
            save_attention_weights=False
        )
    
    def validate(self) -> bool:
        """Validate evaluation configuration"""
        checks = []
        
        # Basic parameter checks
        checks.append(("batch_size > 0", self.batch_size > 0))
        checks.append(("max_samples > 0", self.max_samples > 0))
        checks.append(("generation_num_samples > 0", self.generation_num_samples > 0))
        checks.append(("generation_max_length > 0", self.generation_max_length > 0))
        checks.append(("generation_num_steps > 0", self.generation_num_steps > 0))
        
        # Temperature and sampling checks
        checks.append(("generation_temperature > 0", self.generation_temperature > 0))
        checks.append(("0 < perplexity_masking_rate < 1", 0 < self.perplexity_masking_rate < 1))
        
        # Prompt count checks
        checks.append(("quality_eval_prompts > 0", self.quality_eval_prompts > 0))
        checks.append(("style_eval_prompts > 0", self.style_eval_prompts > 0))
        checks.append(("coherence_eval_prompts > 0", self.coherence_eval_prompts > 0))
        
        # Threshold checks
        checks.append(("excellent_perplexity > 0", self.excellent_perplexity > 0))
        checks.append(("good_perplexity > excellent_perplexity", 
                      self.good_perplexity > self.excellent_perplexity))
        checks.append(("acceptable_perplexity > good_perplexity", 
                      self.acceptable_perplexity > self.good_perplexity))
        
        checks.append(("0 < min_diversity_ratio <= 1", 0 < self.min_diversity_ratio <= 1))
        checks.append(("0 < moderate_diversity_ratio < min_diversity_ratio", 
                      0 < self.moderate_diversity_ratio < self.min_diversity_ratio))
        
        checks.append(("0 <= max_repetition_score < 1", 0 <= self.max_repetition_score < 1))
        checks.append(("max_repetition_score < moderate_repetition_score", 
                      self.max_repetition_score < self.moderate_repetition_score))
        
        # Run all checks
        all_passed = True
        failed_checks = []
        
        for check_name, check_result in checks:
            if not check_result:
                failed_checks.append(check_name)
                all_passed = False
        
        if not all_passed:
            print(f"❌ Evaluation config validation failed: {failed_checks}")
        else:
            print("✅ Evaluation configuration validation passed")
        
        return all_passed
    
    def get_summary(self) -> str:
        """Get human-readable configuration summary"""
        lines = [
            f"Evaluation Configuration Summary:",
            f"  Test Samples: {self.max_samples}",
            f"  Batch Size: {self.batch_size}",
            "",
            f"Generation Settings:",
            f"  Samples per Prompt: {self.generation_num_samples}",
            f"  Max Length: {self.generation_max_length}",
            f"  Diffusion Steps: {self.generation_num_steps}",
            f"  Temperature: {self.generation_temperature}",
            "",
            f"Evaluation Scope:",
            f"  Quality Prompts: {self.quality_eval_prompts}",
            f"  Style Prompts: {self.style_eval_prompts}",
            f"  Coherence Prompts: {self.coherence_eval_prompts}",
            f"  Style References: {self.style_reference_samples}",
            "",
            f"Quality Thresholds:",
            f"  Excellent Perplexity: < {self.excellent_perplexity}",
            f"  Good Perplexity: < {self.good_perplexity}",
            f"  Acceptable Perplexity: < {self.acceptable_perplexity}",
            f"  High Diversity: > {self.min_diversity_ratio:.0%}",
            f"  Low Repetition: < {self.max_repetition_score:.0%}",
            "",
            f"Output Options:",
            f"  Save Samples: {'✅' if self.save_generated_samples else '❌'}",
            f"  Generate Plots: {'✅' if self.generate_plots else '❌'}",
            f"  Detailed Metrics: {'✅' if self.detailed_metrics else '❌'}",
            f"  Memory Efficient: {'✅' if self.memory_efficient_mode else '❌'}"
        ]
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'EvaluationConfig':
        """Create from dictionary"""
        return cls(**config_dict)
    
    def copy(self, **overrides) -> 'EvaluationConfig':
        """Create a copy with optional parameter overrides"""
        config_dict = self.to_dict()
        config_dict.update(overrides)
        return self.from_dict(config_dict)