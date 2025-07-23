"""
Data Pipeline Configuration
All parameters for text processing and curriculum creation
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class PipelineConfig:
    """Configuration for TextDataPipeline"""
    
    # Text segmentation and filtering
    min_sentence_length: int = 20  # Minimum characters
    min_sentence_words: int = 5    # Minimum words
    
    # Lexical difficulty scoring
    lexical_max_word_length: float = 15.0  # Normalization factor
    flesch_score_base: float = 100.0       # Base for Flesch calculation
    sentence_length_max_words: float = 50.0  # Max words for length scoring
    subordinate_clause_max_count: float = 3.0  # Max subordinate clauses
    
    # Syntactic complexity patterns
    subordinate_markers: List[str] = None  # Will be set in __post_init__
    
    # Thematic centrality
    centrality_distance_normalizer: float = 3.0  # Distance normalization
    centrality_max_distance: float = 3.0         # Maximum distance
    
    # Difficulty percentile thresholds  
    easy_difficulty_percentile: float = 33.0   # Bottom 33% = easy
    hard_difficulty_percentile: float = 67.0   # Top 33% = hard
    
    # Vocabulary curriculum levels
    vocab_rare_ratio_thresholds: List[float] = None  # Will be set in __post_init__
    vocab_core_fraction: float = 0.2  # Core vocab = top 20% most frequent
    vocab_level_multiplier: int = 8   # For highest level calculation
    
    # Vocabulary compression
    corpus_coverage_threshold: float = 0.95  # Stop when 95% covered
    min_compressed_vocab_size: int = 1000    # Minimum viable size
    
    # MATTR (Moving Average TTR) calculation
    mattr_window_size: int = 100  # Window size for MATTR
    
    # Dynamic difficulty scoring
    loss_normalization_factor: float = 10.0  # "max reasonable loss ~10"
    gradient_normalization_factor: float = 1.0  # "max grad ~1.0" 
    ema_alpha: float = 0.9  # Exponential moving average factor
    
    # Argument mining confidence thresholds
    min_argument_confidence: float = 0.1
    argument_pattern_bonus: float = 1.0
    question_claim_bonus: float = 0.5
    
    # Readability metrics
    flesch_ease_divisor: float = 100.0
    syllable_estimation_vowels: str = "aeiouy"
    
    # Clustering parameters (from curriculum_config, but pipeline-specific)
    default_n_clusters: int = 8
    embedding_model_default: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Pseudo-data generation limits
    max_pseudo_samples_default: int = 100
    pseudo_data_ratio_default: float = 0.25
    
    def __post_init__(self):
        """Set default values for list fields"""
        if self.subordinate_markers is None:
            self.subordinate_markers = [
                'because', 'since', 'although', 'while', 'if', 
                'when', 'that', 'which', 'who', 'where'
            ]
        
        if self.vocab_rare_ratio_thresholds is None:
            self.vocab_rare_ratio_thresholds = [0.1, 0.3, 0.5]  # Level boundaries
    
    @classmethod
    def default(cls) -> 'PipelineConfig':
        """Default pipeline configuration"""
        return cls()
    
    @classmethod  
    def fast_processing(cls) -> 'PipelineConfig':
        """Faster processing for development/testing"""
        return cls(
            min_sentence_length=10,
            min_sentence_words=3,
            mattr_window_size=50,
            default_n_clusters=4,
            max_pseudo_samples_default=20
        )
    
    @classmethod
    def high_quality(cls) -> 'PipelineConfig':
        """Higher quality processing (slower but better)"""
        return cls(
            min_sentence_length=30,
            min_sentence_words=8,
            mattr_window_size=200,
            default_n_clusters=12,
            corpus_coverage_threshold=0.98,
            min_compressed_vocab_size=2000
        )
    
    def validate(self):
        """Validate pipeline configuration"""
        assert self.min_sentence_length > 0, "min_sentence_length must be positive"
        assert self.min_sentence_words > 0, "min_sentence_words must be positive"
        assert self.lexical_max_word_length > 0, "lexical_max_word_length must be positive"
        assert 0 < self.easy_difficulty_percentile < self.hard_difficulty_percentile < 100, \
            "Invalid difficulty percentiles"
        assert len(self.vocab_rare_ratio_thresholds) > 0, "vocab_rare_ratio_thresholds cannot be empty"
        assert 0 < self.vocab_core_fraction < 1, "vocab_core_fraction must be between 0 and 1"
        assert 0 < self.corpus_coverage_threshold <= 1, "corpus_coverage_threshold must be between 0 and 1"
        assert self.min_compressed_vocab_size > 0, "min_compressed_vocab_size must be positive"
        assert self.mattr_window_size > 0, "mattr_window_size must be positive"
        assert self.loss_normalization_factor > 0, "loss_normalization_factor must be positive"
        assert self.gradient_normalization_factor > 0, "gradient_normalization_factor must be positive"
        assert 0 < self.ema_alpha < 1, "ema_alpha must be between 0 and 1"
        
        return True
