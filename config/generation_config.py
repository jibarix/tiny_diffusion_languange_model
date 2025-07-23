"""
Generation Configuration - ENHANCED VERSION
Text generation parameters for diffusion models
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    """Enhanced text generation configuration"""
    
    # Core sampling parameters
    temperature: float = 0.8
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    
    # Diffusion parameters
    num_steps: int = 50
    confidence_schedule: str = "exponential"  # "linear", "exponential"
    
    # Generation settings
    max_length: int = 512
    min_length: int = 10                      # NEW: Minimum generation length
    num_return_sequences: int = 1
    
    # NEW: Advanced sampling parameters
    confidence_threshold: float = 0.1         # For selective unmasking
    unmasking_schedule: str = "exponential"   # How to decrease masking over steps
    beam_search_width: int = 1                # For beam search generation
    length_penalty: float = 1.0               # Length normalization
    repetition_penalty: float = 1.0           # Prevent repetition
    early_stopping: bool = True               # Stop on EOS token
    seed: Optional[int] = None                # For reproducible generation
    
    # Vocabulary curriculum
    vocab_level: int = 5  # Use highest vocab level by default
    auto_detect_vocab_level: bool = True  # Auto-detect from checkpoint
    
    # Style control
    style_strength: float = 1.0
    style_tokens: list = None
    
    # Interactive settings
    interactive_mode: bool = False
    
    # Output control
    skip_special_tokens: bool = True
    clean_up_tokenization_spaces: bool = True
    
    @classmethod
    def default(cls) -> 'GenerationConfig':
        """Default generation config"""
        return cls()
    
    @classmethod
    def creative(cls) -> 'GenerationConfig':
        """More creative generation"""
        return cls(
            temperature=1.0,
            top_p=0.9,
            num_steps=100,
            repetition_penalty=1.1,
            length_penalty=0.9
        )
    
    @classmethod
    def conservative(cls) -> 'GenerationConfig':
        """More conservative generation"""
        return cls(
            temperature=0.6,
            top_k=50,
            num_steps=30,
            repetition_penalty=1.05,
            early_stopping=True
        )
    
    @classmethod
    def fast(cls) -> 'GenerationConfig':
        """Fast generation with fewer steps"""
        return cls(
            temperature=0.7,
            num_steps=20,
            max_length=256,
            confidence_threshold=0.2  # Less selective for speed
        )
    
    def validate(self):
        """Validate generation configuration"""
        assert 0.1 <= self.temperature <= 2.0, "Temperature must be in [0.1, 2.0]"
        assert self.num_steps > 0, "Number of steps must be positive"
        assert self.max_length > self.min_length, "Max length must exceed min length"
        assert self.num_return_sequences > 0, "Number of sequences must be positive"
        assert 1 <= self.vocab_level <= 5, "Vocab level must be in [1, 5]"
        assert 0 < self.confidence_threshold < 1, "Confidence threshold must be in (0, 1)"
        
        if self.top_k is not None:
            assert self.top_k > 0, "Top-k must be positive"
        
        if self.top_p is not None:
            assert 0 < self.top_p <= 1.0, "Top-p must be in (0, 1]"
        
        assert self.confidence_schedule in ["linear", "exponential"], \
            "Confidence schedule must be 'linear' or 'exponential'"
        
        return True