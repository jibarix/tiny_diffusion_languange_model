"""
Generation Configuration
Text generation parameters for diffusion models
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    """Text generation configuration"""
    
    # Sampling parameters
    temperature: float = 0.8
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    
    # Diffusion parameters
    num_steps: int = 50
    confidence_schedule: str = "exponential"  # "linear", "exponential"
    
    # Generation settings
    max_length: int = 512
    num_return_sequences: int = 1
    
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
            num_steps=100
        )
    
    @classmethod
    def conservative(cls) -> 'GenerationConfig':
        """More conservative generation"""
        return cls(
            temperature=0.6,
            top_k=50,
            num_steps=30
        )
    
    @classmethod
    def fast(cls) -> 'GenerationConfig':
        """Fast generation with fewer steps"""
        return cls(
            temperature=0.7,
            num_steps=20,
            max_length=256
        )
    
    def validate(self):
        """Validate generation configuration"""
        assert 0.1 <= self.temperature <= 2.0, "Temperature must be in [0.1, 2.0]"
        assert self.num_steps > 0, "Number of steps must be positive"
        assert self.max_length > 0, "Max length must be positive"
        assert self.num_return_sequences > 0, "Number of sequences must be positive"
        assert 1 <= self.vocab_level <= 5, "Vocab level must be in [1, 5]"
        
        if self.top_k is not None:
            assert self.top_k > 0, "Top-k must be positive"
        
        if self.top_p is not None:
            assert 0 < self.top_p <= 1.0, "Top-p must be in (0, 1]"
        
        assert self.confidence_schedule in ["linear", "exponential"], \
            "Confidence schedule must be 'linear' or 'exponential'"
        
        return True