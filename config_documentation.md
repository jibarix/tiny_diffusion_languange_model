# Configuration Points

All configuration is managed through a unified system with four main components. Each file contains specific configuration categories:

## Configuration Files Overview

- **`config/__init__.py`** - ProjectConfig (unified manager)
- **`config/model_config.py`** - ModelConfig (architecture parameters)
- **`config/training_config.py`** - TrainingConfig (optimization & training settings)
- **`config/curriculum_config.py`** - CurriculumConfig (curriculum learning stages)

## 1. Project Configuration (`config/__init__.py`)

**ProjectConfig** - Unified configuration manager that imports all other configs

### Project Paths
```python
data_dir: str = "data"              # Data directory path
output_dir: str = "outputs"         # Output directory for checkpoints/logs
cache_dir: str = "cache"            # Cache directory for processed data
```

### Hardware Settings
```python
device: str = "cuda"                # Device: "cuda" or "cpu"
mixed_precision: bool = True        # Enable mixed precision training
gradient_checkpointing: bool = True # Enable gradient checkpointing for memory
```

### Experiment Settings
```python
experiment_name: str = "micro-diffusion"  # Experiment name for outputs
seed: int = 42                           # Random seed for reproducibility
```

### Usage
```python
# Load from YAML
config = ProjectConfig.from_yaml("config.yaml")

# Create default
config = ProjectConfig.default()

# Save to YAML
config.save_yaml("config.yaml")
```

## 2. Model Configuration (`config/model_config.py`)

**ModelConfig** - Transformer architecture parameters

### Core Architecture
```python
d_model: int = 768          # Model dimension (768 for 125M params)
n_layers: int = 12          # Number of transformer layers
n_heads: int = 12           # Number of attention heads
d_ff: int = 2048           # Feed-forward dimension
```

### Vocabulary & Sequence
```python
vocab_size: int = 25000     # Vocabulary size (compressed from 50k GPT-2)
max_seq_len: int = 512      # Maximum sequence length
```

### Regularization
```python
dropout: float = 0.1           # General dropout rate
attention_dropout: float = 0.1  # Attention-specific dropout
```

### Architecture Components
```python
activation: str = "swiglu"     # Activation function: "swiglu", "gelu", "relu"
norm_type: str = "rmsnorm"     # Normalization: "layernorm", "rmsnorm"
norm_eps: float = 1e-6         # Normalization epsilon
pos_encoding: str = "rope"     # Position encoding: "rope", "learned", "sinusoidal"
```

### Initialization & Efficiency
```python
init_std: float = 0.02         # Initialization standard deviation
use_bias: bool = False         # Remove bias terms for efficiency
mask_token_id: int = 0         # Mask token ID (set during tokenizer creation)
```

### Predefined Configurations
```python
# ~125M parameters
ModelConfig.tiny_125m()

# ~350M parameters  
ModelConfig.small_350m()
```

### Parameter Count Formula
```python
# P = 4lh² + 3lh·hf + 6lh + Vh
# l=layers, h=d_model, hf=d_ff, V=vocab_size
config.param_count  # Automatic calculation
```

## 3. Training Configuration (`config/training_config.py`)

**TrainingConfig** - Optimization and training hyperparameters

### Optimization
```python
learning_rate: float = 2e-4      # Initial learning rate
min_learning_rate: float = 2e-5  # Minimum learning rate for scheduling
weight_decay: float = 0.1        # L2 regularization
beta1: float = 0.9              # Adam beta1 parameter
beta2: float = 0.95             # Adam beta2 parameter
eps: float = 1e-8               # Adam epsilon
grad_clip_norm: float = 1.0     # Gradient clipping norm
```

### Learning Rate Schedule
```python
warmup_steps: int = 1000                      # Warmup steps
scheduler: str = "cosine_with_restarts"       # "cosine", "linear", "constant"
```

### Batch Settings
```python
batch_size: int = 32                    # Training batch size
gradient_accumulation_steps: int = 1    # Gradient accumulation steps
max_grad_norm: float = 1.0             # Maximum gradient norm
```

### Training Length
```python
max_epochs: int = 300           # Maximum training epochs
max_steps: Optional[int] = None # Maximum training steps (overrides epochs)
```

### Evaluation & Checkpointing
```python
eval_every: int = 1000         # Evaluate every N steps
save_every: int = 5000         # Save checkpoint every N steps
log_every: int = 100           # Log metrics every N steps
```

### Early Stopping
```python
patience: int = 10             # Early stopping patience (epochs)
min_delta: float = 1e-4        # Minimum improvement threshold
```

### Validation
```python
val_split: float = 0.1                  # Validation split ratio
val_batch_size: Optional[int] = None    # Validation batch size (uses batch_size if None)
```

### Memory Optimization
```python
use_gradient_checkpointing: bool = True  # Enable gradient checkpointing
use_mixed_precision: bool = True         # Enable mixed precision training
dataloader_num_workers: int = 4          # DataLoader worker processes
pin_memory: bool = True                  # Pin memory for faster GPU transfer
```

### Regularization
```python
label_smoothing: float = 0.0        # Label smoothing factor
dropout_schedule: str = "constant"   # Dropout scheduling: "constant", "decay"
```

### Computed Properties
```python
config.effective_batch_size      # batch_size * gradient_accumulation_steps
config.val_batch_size_actual     # Actual validation batch size
```

### Predefined Configurations
```python
# Default for 8GB VRAM
TrainingConfig.default()

# Memory-efficient for limited VRAM
TrainingConfig.memory_efficient()

# Fast debugging
TrainingConfig.fast_debug()
```

## 4. Curriculum Configuration (`config/curriculum_config.py`)

**CurriculumConfig** - Three-stage curriculum learning setup

### Stage Definition (`StageConfig`)
```python
class StageConfig:
    name: str                           # Stage name
    epochs: int                         # Number of epochs for this stage
    masking_rate_range: Tuple[float, float]  # (min, max) masking rate
    data_selection: str                 # "easy", "medium", "hard", "all"
    format_type: str                    # "sentences", "pairs", "paragraphs"
```

### Curriculum Stages
```python
stages: List[StageConfig]  # List of curriculum stages
```

### Difficulty Scoring Weights
```python
lexical_weight: float = 0.3      # Weight for lexical rarity scoring
syntactic_weight: float = 0.3    # Weight for syntactic complexity
centrality_weight: float = 0.4   # Weight for thematic centrality
```

### Data Selection Thresholds
```python
easy_threshold: float = 33.0     # Bottom 33% complexity (percentile)
hard_threshold: float = 67.0     # Top 33% complexity (percentile)
```

### Clustering Parameters
```python
n_clusters: int = 8                                      # Number of clusters for data
embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
```

### Transition Settings
```python
transition_epochs: int = 5       # Gradual transition between stages
reset_optimizer: bool = True     # Reset optimizer between stages
```

### Pseudo Data Generation (Stage 3)
```python
pseudo_data_max_samples: int = 100  # Maximum pseudo samples to generate
pseudo_data_ratio: float = 0.25     # Ratio of pseudo to real data (1/4)
```

### Predefined Curriculum Schedules

#### Three-Stage Curriculum (Default)
```python
CurriculumConfig.three_stage()
# Stage 1: Foundation (50 epochs, 75-90% masking, easy data, sentences)
# Stage 2: Structural (100 epochs, 40-60% masking, medium data, pairs)  
# Stage 3: Refinement (150 epochs, 10-30% masking, all data, paragraphs)
```

#### Single-Stage Baseline
```python
CurriculumConfig.single_stage()
# Single stage: 300 epochs, 10-90% masking, all data, sentences
```

#### Fast Debug
```python
CurriculumConfig.fast_debug()
# Stage 1: 2 epochs, 75-90% masking, easy data
# Stage 2: 3 epochs, 10-30% masking, all data
```

### Stage Properties
```python
stage.masking_rate_min    # Minimum masking rate
stage.masking_rate_max    # Maximum masking rate
```

### Curriculum Properties
```python
config.total_epochs          # Total epochs across all stages
config.stage_names          # List of stage names
config.get_stage_by_name(name)    # Get stage by name
config.get_stage_by_epoch(epoch)  # Get stage for given epoch
```

## Configuration Loading Examples

### Loading from YAML
```python
from config import ProjectConfig

# Load complete configuration
config = ProjectConfig.from_yaml("config.yaml")

# Access sub-configs
model_config = config.model
training_config = config.training
curriculum_config = config.curriculum
```

### Creating Custom Configurations
```python
from config import ProjectConfig, ModelConfig, TrainingConfig, CurriculumConfig

# Custom model for different size
custom_model = ModelConfig(
    d_model=512,
    n_layers=8,
    n_heads=8,
    d_ff=1536,
    vocab_size=20000
)

# Memory-efficient training
memory_training = TrainingConfig.memory_efficient()

# Custom curriculum
custom_curriculum = CurriculumConfig(
    stages=[
        StageConfig(
            name="custom_stage",
            epochs=100,
            masking_rate_range=(0.5, 0.8),
            data_selection="medium",
            format_type="sentences"
        )
    ]
)

# Combine into project config
config = ProjectConfig(
    model=custom_model,
    training=memory_training,
    curriculum=custom_curriculum,
    experiment_name="custom_experiment"
)
```

### Validation
All configuration classes include validation methods:
```python
config.model.validate()                    # Check model architecture validity
config.training.validate(config.model)    # Check training hyperparameters with memory estimation
config.curriculum.validate()              # Check curriculum setup
```

## 5. Generation Configuration (MISSING - Needs Implementation)

**GenerationConfig** - Text generation parameters currently hardcoded in scripts

### Sampling Parameters
```python
temperature: float = 0.8        # Sampling temperature (currently hardcoded in scripts/generate.py)
top_k: Optional[int] = None     # Top-k sampling
top_p: Optional[float] = None   # Nucleus sampling
num_steps: int = 50            # Diffusion denoising steps
max_length: int = 512          # Maximum generation length
num_return_sequences: int = 1   # Number of sequences to generate
```

### Generation Strategy
```python
confidence_schedule: str = "linear"    # Confidence scheduling for unmasking
style_strength: float = 1.0           # Style token boosting strength
interactive_mode: bool = False        # Enable interactive generation
```

## 6. Data Processing Configuration (MISSING - Needs Implementation)

**DataConfig** - Data preprocessing parameters currently hardcoded

### Text Processing
```python
min_sentence_length: int = 10         # Minimum sentence length
max_sentence_length: int = 512        # Maximum sentence length  
sentence_overlap: int = 50            # Overlap for sentence pairs
paragraph_max_sentences: int = 10     # Max sentences per paragraph
```

### Difficulty Scoring
```python
idf_corpus_size: int = 100000        # Corpus size for IDF calculation
readability_metrics: List[str] = ["flesch", "coleman_liau", "automated_readability"]
complexity_weights: Dict[str, float] = {
    "lexical": 0.3,
    "syntactic": 0.3, 
    "centrality": 0.4
}
```

## 7. Hardware Configuration (PARTIALLY MISSING)

**HardwareConfig** - Memory and performance settings

### Memory Management
```python
max_vram_gb: float = 8.0             # VRAM limit (currently hardcoded as 8GB)
memory_safety_margin: float = 0.1    # Safety margin for memory estimation
bytes_per_param: int = 4             # Bytes per parameter (currently hardcoded)
memory_multiplier: float = 3.0       # Memory multiplier for model+optimizer+gradients
```

## Current Hardcoded Values That Need Configuration

### In `config/training_config.py` - Line 79-82:
```python
# HARDCODED - Should be configurable
model_params_mb = 125  # ~125M parameters
batch_mem_mb = self.batch_size * 512 * 4 / 1e6  # 4 bytes per token
estimated_vram_gb = (model_params_mb * 3 + batch_mem_mb) / 1000  # 3x multiplier
if estimated_vram_gb > 8:  # 8GB limit hardcoded
```

### In `scripts/generate.py` - Line 59-63:
```python
# HARDCODED - Should use GenerationConfig
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--num-steps", type=int, default=50) 
parser.add_argument("--max-length", type=int, default=512)
```

### In `src/evaluation/generate.py`:
```python
# HARDCODED constants throughout generation logic
confidence_schedule = self._get_confidence_schedule(step, total_steps)  # Uses hardcoded schedule
```

## Recommended Configuration Structure

```python
# Complete configuration with all missing pieces
@dataclass
class ProjectConfig:
    model: ModelConfig
    training: TrainingConfig  
    curriculum: CurriculumConfig
    generation: GenerationConfig      # MISSING - needs implementation
    data: DataConfig                  # MISSING - needs implementation  
    hardware: HardwareConfig          # PARTIALLY MISSING
    
    # Project paths (already implemented)
    data_dir: str = "data"
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    
    # Experiment settings (already implemented)
    experiment_name: str = "micro-diffusion"
    seed: int = 42
```

## YAML Configuration File Example

```yaml
# Project settings
experiment_name: "my-diffusion-model"
data_dir: "data"
output_dir: "outputs"
device: "cuda"
mixed_precision: true
seed: 42

# Hardware constraints
hardware:
  max_vram_gb: 8.0
  memory_safety_margin: 0.1
  bytes_per_param: 4
  memory_multiplier: 3.0

# Model architecture
model:
  d_model: 768
  n_layers: 12
  n_heads: 12
  d_ff: 2048
  vocab_size: 25000
  max_seq_len: 512
  dropout: 0.1

# Training hyperparameters
training:
  learning_rate: 0.0002
  batch_size: 32
  max_epochs: 300
  eval_every: 1000
  save_every: 5000

# Text generation
generation:
  temperature: 0.8
  num_steps: 50
  max_length: 512
  top_k: null
  top_p: null
  confidence_schedule: "linear"

# Data processing
data:
  min_sentence_length: 10
  max_sentence_length: 512
  sentence_overlap: 50
  idf_corpus_size: 100000

# Curriculum learning
curriculum:
  stages:
    - name: "foundation"
      epochs: 50
      masking_rate_range: [0.75, 0.90]
      data_selection: "easy"
      format_type: "sentences"
    - name: "structural"
      epochs: 100
      masking_rate_range: [0.40, 0.60]
      data_selection: "medium"
      format_type: "pairs"
    - name: "refinement"
      epochs: 150
      masking_rate_range: [0.10, 0.30]
      data_selection: "all"
      format_type: "paragraphs"
```