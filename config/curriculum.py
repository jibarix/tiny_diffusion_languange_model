"""
Curriculum Learning Configuration

Defines the 3-stage curriculum learning framework for masked diffusion training:
Stage I: Foundational (high masking, simple sentences, central themes)
Stage II: Structural (medium masking, argument pairs, logical flow)  
Stage III: Refinement (low masking, full complexity, long passages)

Based on 2025 research on curriculum learning for small language models.
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage"""
    name: str
    epochs: int
    masking_rate_range: Tuple[float, float]  # (min, max) masking rates
    data_selection_criteria: Dict[str, Any]
    training_format: str  # 'sentences', 'pairs', 'paragraphs'
    description: str


def get_curriculum_config() -> Dict[str, Any]:
    """
    Get 3-stage curriculum configuration for masked diffusion training.
    
    Returns comprehensive curriculum with data selection criteria,
    masking schedules, and format specifications for each stage.
    """
    
    stages = [
        # Stage I: Foundational Learning
        CurriculumStage(
            name="foundational",
            epochs=50,
            masking_rate_range=(0.75, 0.90),
            data_selection_criteria={
                'syntactic_complexity': 'bottom_33_percent',  # Simple sentences
                'lexical_rarity': 'bottom_33_percent',        # Common vocabulary
                'thematic_centrality': 'top_33_percent',      # Prototypical examples
                'min_sentence_length': 10,
                'max_sentence_length': 50,
                'exclude_dialogue': True,
                'exclude_citations': True,
            },
            training_format='sentences',
            description="Learn core vocabulary, basic syntax, central themes"
        ),
        
        # Stage II: Structural Learning  
        CurriculumStage(
            name="structural",
            epochs=100,
            masking_rate_range=(0.40, 0.60),
            data_selection_criteria={
                'syntactic_complexity': 'bottom_66_percent',  # Easy to moderate
                'lexical_rarity': 'bottom_66_percent',
                'argument_structure': ['evidence', 'claim', 'warrant'],  # Logical components
                'min_sentence_length': 15,
                'max_sentence_length': 100,
                'require_logical_pairs': True,
            },
            training_format='pairs', 
            description="Learn argumentative relationships and logical flow"
        ),
        
        # Stage III: Refinement
        CurriculumStage(
            name="refinement", 
            epochs=300,
            masking_rate_range=(0.05, 0.20),
            data_selection_criteria={
                'use_full_corpus': True,  # All data including complex examples
                'include_outliers': True,
                'min_sentence_length': 5,
                'max_sentence_length': 200,
                'enable_self_training': True,  # Add generated pseudo-data
            },
            training_format='paragraphs',
            description="Master full complexity and generate coherent passages"
        )
    ]
    
    return {
        'stages': [
            {
                'name': stage.name,
                'epochs': stage.epochs,
                'masking_rate_range': stage.masking_rate_range,
                'data_selection_criteria': stage.data_selection_criteria,
                'training_format': stage.training_format,
                'description': stage.description,
            }
            for stage in stages
        ],
        
        # Global curriculum settings
        'transition_strategy': 'gradual',  # 'abrupt' or 'gradual'
        'reset_optimizer_between_stages': False,
        'adjust_learning_rate_between_stages': True,
        'learning_rate_decay_factors': [1.0, 0.8, 0.6],  # Per stage
        
        # Data scoring parameters
        'difficulty_scoring': {
            'lexical_rarity_corpus': 'wikipedia',  # Reference corpus for IDF
            'centrality_clustering_method': 'kmeans',
            'num_clusters': 10,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'syntactic_features': [
                'sentence_length',
                'num_clauses', 
                'flesch_kincaid_score',
                'num_rare_words'
            ]
        },
        
        # Masking schedule details
        'masking_schedule': {
            'schedule_type': 'linear',  # 'linear', 'cosine', 'constant'
            'warmup_steps': 100,        # Steps to reach target masking rate
            'dynamic_adjustment': False, # Adjust based on loss
        },
        
        # Self-training for Stage III
        'self_training': {
            'enabled': True,
            'generation_interval': 1000,    # Generate pseudo-data every N steps
            'quality_threshold': 0.8,       # Similarity to original data
            'max_pseudo_data_ratio': 0.2,   # Max 20% pseudo-data
            'filtering_method': 'semantic_similarity',
        },
        
        # Evaluation during curriculum
        'curriculum_evaluation': {
            'eval_between_stages': True,
            'stage_completion_criteria': 'epochs',  # 'epochs' or 'loss_plateau'
            'loss_plateau_patience': 5,
            'min_improvement_threshold': 0.01,
        }
    }


def get_difficulty_scoring_config() -> Dict[str, Any]:
    """Configuration for multi-dimensional difficulty scoring"""
    return {
        'lexical_difficulty': {
            'method': 'idf_based',
            'reference_corpus': 'wikipedia',
            'rare_word_threshold': 1e-5,  # Words with freq < threshold are rare
            'weight': 0.3,
        },
        
        'syntactic_difficulty': {
            'method': 'composite_score',
            'features': {
                'sentence_length': {'weight': 0.3, 'normalize': True},
                'num_subordinate_clauses': {'weight': 0.2, 'normalize': True},
                'flesch_kincaid_grade': {'weight': 0.3, 'normalize': True},
                'parse_tree_depth': {'weight': 0.2, 'normalize': True},
            },
            'weight': 0.4,
        },
        
        'thematic_centrality': {
            'method': 'cluster_distance',
            'clustering_algorithm': 'kmeans',
            'num_clusters': 10,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'distance_metric': 'euclidean',
            'weight': 0.3,
        },
        
        'composite_scoring': {
            'aggregation_method': 'weighted_sum',  # 'weighted_sum' or 'rank_fusion'
            'normalization': 'min_max',           # 'min_max' or 'z_score'
        }
    }


def get_argument_structure_config() -> Dict[str, Any]:
    """Configuration for argumentative structure parsing (optional for non-argumentative texts)"""
    return {
        'enabled': True,  # Set to False for pure narrative texts
        'framework': 'toulmin',  # 'toulmin' or 'simple'
        
        'toulmin_components': [
            'claim',      # Central assertion
            'evidence',   # Supporting data/facts
            'warrant',    # Logical bridge
            'backing',    # Support for warrant
            'rebuttal',   # Counter-arguments
            'conclusion', # Summary statement
        ],
        
        'simple_components': [
            'premise',    # Supporting statement
            'conclusion', # Main point
            'example',    # Illustrative case
        ],
        
        'parsing_method': 'rule_based',  # 'rule_based' or 'ml_based'
        'confidence_threshold': 0.7,
        
        # Rule-based patterns (simplified)
        'argument_patterns': {
            'evidence_indicators': ['for example', 'research shows', 'data indicates'],
            'claim_indicators': ['therefore', 'thus', 'in conclusion'],
            'warrant_indicators': ['because', 'since', 'given that'],
            'rebuttal_indicators': ['however', 'although', 'critics argue'],
        }
    }


def validate_curriculum_config(config: Dict[str, Any]) -> None:
    """Validate curriculum configuration parameters"""
    stages = config['stages']
    
    # Check stage count
    assert len(stages) == 3, f"Expected 3 stages, got {len(stages)}"
    
    # Validate each stage
    for i, stage in enumerate(stages):
        # Check masking rate progression (should decrease)
        min_mask, max_mask = stage['masking_rate_range']
        assert 0 <= min_mask <= max_mask <= 1, f"Invalid masking range in stage {i}"
        
        if i > 0:
            prev_min, prev_max = stages[i-1]['masking_rate_range']
            assert max_mask < prev_min, f"Masking rate should decrease across stages"
        
        # Check epochs
        assert stage['epochs'] > 0, f"Stage {i} epochs must be positive"
        
        # Validate format
        valid_formats = ['sentences', 'pairs', 'paragraphs']
        assert stage['training_format'] in valid_formats, \
            f"Invalid format '{stage['training_format']}' in stage {i}"
    
    # Check total epochs vs difficulty progression
    total_epochs = sum(stage['epochs'] for stage in stages)
    assert total_epochs > 0, "Total curriculum epochs must be positive"
    
    print(f"Curriculum validation passed:")
    print(f"  Total epochs: {total_epochs}")
    print(f"  Masking progression: {[s['masking_rate_range'] for s in stages]}")


def get_curriculum_presets() -> Dict[str, Dict[str, Any]]:
    """Predefined curriculum configurations for different scenarios"""
    return {
        'default': get_curriculum_config(),
        
        'debug': {
            'stages': [
                {
                    'name': 'foundational',
                    'epochs': 1,
                    'masking_rate_range': (0.8, 0.9),
                    'data_selection_criteria': {'use_first_n_sentences': 100},
                    'training_format': 'sentences',
                    'description': 'Debug foundational stage'
                },
                {
                    'name': 'structural', 
                    'epochs': 1,
                    'masking_rate_range': (0.5, 0.6),
                    'data_selection_criteria': {'use_first_n_sentences': 50},
                    'training_format': 'pairs',
                    'description': 'Debug structural stage'
                },
                {
                    'name': 'refinement',
                    'epochs': 1, 
                    'masking_rate_range': (0.2, 0.3),
                    'data_selection_criteria': {'use_first_n_sentences': 25},
                    'training_format': 'paragraphs',
                    'description': 'Debug refinement stage'
                }
            ],
            'transition_strategy': 'abrupt',
            'difficulty_scoring': {'enabled': False},  # Skip complex scoring for debug
        },
        
        'fast': {
            'stages': [
                {
                    'name': 'foundational',
                    'epochs': 10,
                    'masking_rate_range': (0.7, 0.8),
                    'data_selection_criteria': {'syntactic_complexity': 'bottom_50_percent'},
                    'training_format': 'sentences',
                    'description': 'Fast foundational stage'
                },
                {
                    'name': 'structural',
                    'epochs': 20,
                    'masking_rate_range': (0.4, 0.5),
                    'data_selection_criteria': {'syntactic_complexity': 'bottom_75_percent'},
                    'training_format': 'pairs', 
                    'description': 'Fast structural stage'
                },
                {
                    'name': 'refinement',
                    'epochs': 30,
                    'masking_rate_range': (0.1, 0.2),
                    'data_selection_criteria': {'use_full_corpus': True},
                    'training_format': 'paragraphs',
                    'description': 'Fast refinement stage'
                }
            ]
        },
        
        'narrative_only': {
            # For pure narrative texts without argumentative structure
            'stages': [
                {
                    'name': 'foundational',
                    'epochs': 40,
                    'masking_rate_range': (0.75, 0.85),
                    'data_selection_criteria': {
                        'syntactic_complexity': 'bottom_33_percent',
                        'dialogue_ratio': 'low',  # Start with narrative
                    },
                    'training_format': 'sentences',
                    'description': 'Learn basic narrative structure'
                },
                {
                    'name': 'character_dialogue',
                    'epochs': 80,
                    'masking_rate_range': (0.4, 0.6),
                    'data_selection_criteria': {
                        'include_dialogue': True,
                        'character_interactions': True,
                    },
                    'training_format': 'pairs',
                    'description': 'Learn character interactions and dialogue'
                },
                {
                    'name': 'full_narrative',
                    'epochs': 120,
                    'masking_rate_range': (0.15, 0.35),
                    'data_selection_criteria': {'use_full_corpus': True},
                    'training_format': 'paragraphs',
                    'description': 'Master complete narrative generation'
                }
            ],
            'argument_structure': {'enabled': False},
        }
    }


def print_curriculum_summary(config: Dict[str, Any]) -> None:
    """Print comprehensive curriculum configuration summary"""
    stages = config['stages']
    
    print("=" * 60)
    print("CURRICULUM LEARNING CONFIGURATION")
    print("=" * 60)
    
    total_epochs = sum(stage['epochs'] for stage in stages)
    print(f"Total Training Epochs: {total_epochs}")
    print(f"Transition Strategy: {config.get('transition_strategy', 'gradual')}")
    
    print(f"\nStage Breakdown:")
    for i, stage in enumerate(stages, 1):
        min_mask, max_mask = stage['masking_rate_range']
        print(f"\n  Stage {i}: {stage['name'].title()}")
        print(f"    Epochs: {stage['epochs']} ({stage['epochs']/total_epochs:.1%} of total)")
        print(f"    Masking Rate: {min_mask:.1%} - {max_mask:.1%}")
        print(f"    Format: {stage['training_format']}")
        print(f"    Goal: {stage['description']}")
    
    print(f"\nDifficulty Scoring:")
    scoring = config.get('difficulty_scoring', {})
    if scoring.get('enabled', True):
        print(f"  Clustering Method: {scoring.get('clustering_method', 'kmeans')}")
        print(f"  Number of Clusters: {scoring.get('num_clusters', 10)}")
        print(f"  Embedding Model: {scoring.get('embedding_model', 'sentence-transformers')}")
    else:
        print(f"  Disabled (using simple heuristics)")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test curriculum configurations
    print("Testing curriculum configurations...")
    
    # Test default config
    config = get_curriculum_config()
    validate_curriculum_config(config)
    print_curriculum_summary(config)
    
    # Test presets
    presets = get_curriculum_presets()
    for preset_name in ['debug', 'fast']:
        print(f"\n{preset_name.upper()} PRESET:")
        preset_config = presets[preset_name]
        validate_curriculum_config(preset_config)
        
        total_epochs = sum(stage['epochs'] for stage in preset_config['stages'])
        masking_ranges = [stage['masking_rate_range'] for stage in preset_config['stages']]
        print(f"  Total epochs: {total_epochs}")
        print(f"  Masking progression: {masking_ranges}")