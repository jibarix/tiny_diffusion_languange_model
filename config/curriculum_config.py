"""
Curriculum Learning Configuration
Defines three-stage learning progression
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List


@dataclass
class StageConfig:
    """Configuration for a single curriculum stage"""
    name: str
    epochs: int
    masking_rate_range: Tuple[float, float]  # (min, max) masking rate
    data_selection: str  # "easy", "medium", "hard", "all"
    format_type: str  # "sentences", "pairs", "paragraphs"
    
    @property
    def masking_rate_min(self) -> float:
        return self.masking_rate_range[0]
    
    @property 
    def masking_rate_max(self) -> float:
        return self.masking_rate_range[1]


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration"""
    
    # Stage definitions
    stages: List[StageConfig]
    
    # Difficulty scoring weights
    lexical_weight: float = 0.3
    syntactic_weight: float = 0.3  
    centrality_weight: float = 0.4
    
    # Data selection thresholds (percentiles)
    easy_threshold: float = 33.0  # Bottom 33% complexity
    hard_threshold: float = 67.0  # Top 33% complexity
    
    # Clustering parameters
    n_clusters: int = 8
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Transition settings
    transition_epochs: int = 5  # Gradual transition between stages
    reset_optimizer: bool = True  # Reset optimizer between stages
    
    @classmethod
    def three_stage(cls) -> 'CurriculumConfig':
        """Default three-stage curriculum for Micro Diffusion project"""
        stages = [
            StageConfig(
                name="foundation",
                epochs=50,
                masking_rate_range=(0.75, 0.90),
                data_selection="easy",
                format_type="sentences"
            ),
            StageConfig(
                name="structural", 
                epochs=100,
                masking_rate_range=(0.40, 0.60),
                data_selection="medium",
                format_type="pairs"
            ),
            StageConfig(
                name="refinement",
                epochs=150,
                masking_rate_range=(0.10, 0.30), 
                data_selection="all",
                format_type="paragraphs"
            )
        ]
        
        return cls(stages=stages)
    
    @classmethod
    def single_stage(cls) -> 'CurriculumConfig':
        """Single stage for baseline comparison"""
        stages = [
            StageConfig(
                name="baseline",
                epochs=300,
                masking_rate_range=(0.10, 0.90),
                data_selection="all", 
                format_type="sentences"
            )
        ]
        
        return cls(stages=stages)
    
    @classmethod
    def fast_debug(cls) -> 'CurriculumConfig':
        """Quick curriculum for debugging"""
        stages = [
            StageConfig(
                name="debug_foundation",
                epochs=2,
                masking_rate_range=(0.75, 0.90),
                data_selection="easy",
                format_type="sentences"
            ),
            StageConfig(
                name="debug_refinement", 
                epochs=3,
                masking_rate_range=(0.10, 0.30),
                data_selection="all",
                format_type="sentences"
            )
        ]
        
        return cls(stages=stages)
    
    @property
    def total_epochs(self) -> int:
        """Total epochs across all stages"""
        return sum(stage.epochs for stage in self.stages)
    
    @property
    def stage_names(self) -> List[str]:
        """List of stage names"""
        return [stage.name for stage in self.stages]
    
    def get_stage_by_name(self, name: str) -> StageConfig:
        """Get stage configuration by name"""
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise ValueError(f"Stage '{name}' not found")
    
    def get_stage_by_epoch(self, epoch: int) -> Tuple[int, StageConfig]:
        """Get stage index and config for given epoch"""
        cumulative = 0
        for i, stage in enumerate(self.stages):
            if epoch < cumulative + stage.epochs:
                return i, stage
            cumulative += stage.epochs
        
        # Return last stage if epoch exceeds total
        return len(self.stages) - 1, self.stages[-1]
    
    def validate(self):
        """Validate curriculum configuration"""
        assert len(self.stages) > 0, "Must have at least one stage"
        assert 0 < self.easy_threshold < self.hard_threshold < 100, "Invalid thresholds"
        
        total_weight = self.lexical_weight + self.syntactic_weight + self.centrality_weight
        assert abs(total_weight - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total_weight}"
        
        for stage in self.stages:
            assert 0 <= stage.masking_rate_min <= stage.masking_rate_max <= 1, \
                f"Invalid masking rates for stage {stage.name}"
        
        return True