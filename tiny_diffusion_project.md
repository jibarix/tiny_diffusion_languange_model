# Tiny Text Diffusion Model: Single-Book Style Generator

## Project Overview

A fun exploration of training a small masked diffusion language model on any single book using consumer hardware. This project implements curriculum learning strategies to maximize performance in data-constrained settings.

### Key Insights from Research
- **Diffusion beats autoregressive in data-constrained settings**: When data is limited but compute is available, masked diffusion models significantly outperform AR models
- **Critical compute threshold**: Diffusion becomes favorable when compute exceeds `Ccrit(U) = 2.12 × 10^15 · U^2.174` FLOPs
- **Superior data reuse**: Diffusion models can benefit from ~500 epochs vs ~15 for AR models

## Hardware Specifications

```
CPU: Intel i7-12700H (14 cores, 20 threads)
GPU: RTX 3070 Ti Laptop (8GB VRAM)
RAM: 64GB DDR5-4800
```

## Architecture Design

### Model Configuration (Target: ~125M parameters)
- **Architecture**: Transformer with bidirectional attention (no causal masking)
- **Layers**: 12 (following "deeper not wider" principle for tiny models)
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Vocab Size**: ~25k tokens (compressed from 50k GPT-2 vocab using single book corpus frequency analysis)
- **Sequence Length**: 512 tokens
- **Positional Encoding**: RoPE (Rotary Position Embeddings)

### Architectural Choices Rationale
- **Compressed Tokenizer**: Reduces embedding parameters from 36.8% to <20% of total params
- **Deeper Architecture**: 12 layers provide better performance than wider alternatives for small models
- **Bidirectional Attention**: Essential for masked diffusion training

## Training Strategy: Generative Stylography Framework

### Three-Stage Curriculum Learning

#### Stage I: Foundational (Epochs 1-50)
- **Objective**: Learn core vocabulary and basic sentence structures
- **Data**: High-centrality, low-complexity sentences (bottom 33% syntactic complexity)
- **Masking Rate**: 75-90% (easy denoising task)
- **Format**: Individual sentences

#### Stage II: Structural (Epochs 51-150)
- **Objective**: Learn argumentative relationships and logical flow
- **Data**: Evidence-claim pairs, moderate complexity
- **Masking Rate**: 40-60% (medium difficulty)
- **Format**: Structured pairs `<Evidence> [SEP] <Claim>`

#### Stage III: Refinement (Epochs 151-300+)
- **Objective**: Master full complexity and generate coherent passages
- **Data**: Full corpus including complex sentences
- **Masking Rate**: 10-30% (hard denoising)
- **Format**: Full paragraphs (up to 512 tokens)

### Data Preparation Pipeline

#### 1. Text Preprocessing
```python
# Segment any book into sentences
# Calculate difficulty scores:
difficulty_scores = {
    'lexical_rarity': calculate_idf_scores(sentences, general_corpus),
    'syntactic_complexity': calculate_readability_scores(sentences),
    'argument_structure': parse_argumentative_roles(sentences),  # Optional for narrative
    'thematic_centrality': calculate_cluster_centrality(sentence_embeddings)
}
```

#### 2. Curriculum Construction
- **Easy Examples**: High centrality + low complexity + common vocabulary
- **Medium Examples**: Moderate complexity + clear argumentative structure
- **Hard Examples**: All data including outliers and complex arguments

## Implementation Plan

### Phase 1: Data Pipeline
- [x] Load and clean any book text
- [x] Implement tokenizer compression based on book vocabulary
- [x] Create difficulty scoring system
- [x] Generate curriculum schedules for all three stages

### Phase 2: Model Implementation
- [x] Implement masked diffusion transformer in PyTorch
- [x] Create training loop with curriculum scheduling
- [x] Implement masking strategies (uniform random masking)
- [x] Add logging and checkpointing

### Phase 3: Training
- [ ] Stage I training (50 epochs)
- [ ] Stage II training (100 epochs)
- [ ] Stage III training (150+ epochs)
- [ ] Monitor for convergence and adjust as needed

### Phase 4: Evaluation
- [x] Perplexity on held-out text
- [x] Style similarity analysis (sentence length distribution, vocabulary usage)
- [x] Human evaluation of generated samples
- [x] Comparison with baseline autoregressive model

## Training Configuration

```python
# Hyperparameters (based on Muennighoff et al. 2024)
BATCH_SIZE = 32  # Fits in 8GB VRAM with gradient checkpointing
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 1000
GRADIENT_CLIPPING = 1.0
OPTIMIZER = "AdamW"
SCHEDULER = "cosine_with_restarts"

# Curriculum parameters
MASKING_SCHEDULES = {
    'stage_1': (0.75, 0.90),  # High masking rate
    'stage_2': (0.40, 0.60),  # Medium masking rate  
    'stage_3': (0.10, 0.30)   # Low masking rate
}
```

## Expected Outcomes

### Performance Targets
- **Perplexity**: <15 on held-out text (baseline AR model likely ~20)
- **Style Fidelity**: Generated text matches author's sentence length distribution
- **Coherence**: Logical flow in generated passages
- **Novelty**: Generate new content, not memorized passages

### Learning Trajectory
Based on research, expect:
- Stage I: Rapid initial learning, plateau around epoch 30
- Stage II: Steady improvement in logical structure
- Stage III: Gradual refinement, potential for 200+ beneficial epochs

## File Structure

```
tiny-diffusion/
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py           # Unified ProjectConfig namespace
│   ├── model_config.py       # Architecture parameters (125M+ configurable)
│   ├── training_config.py    # Training hyperparameters + hardware config
│   ├── curriculum_config.py  # 3-stage learning schedule
│   ├── generation_config.py  # Text generation parameters
│   ├── pipeline_config.py    # Data processing parameters
│   └── evaluation_config.py  # Evaluation & benchmarking config
├── data/
│   ├── raw/
│   │   └── [your_book].txt
│   ├── processed/
│   │   ├── segments.pkl
│   │   ├── curriculum_splits.pkl
│   │   ├── vocab_curriculum.pkl
│   │   └── dynamic_scorer.pkl
│   └── tokenizer/             # Or tokenizer_level_1/ through tokenizer_level_5/
│       └── compressed_tokenizer.json
├── src/
│   ├── model/
│   │   ├── attention.py      # Multi-head attention with RoPE
│   │   ├── transformer.py    # Transformer blocks with SwiGLU
│   │   └── diffusion.py      # Masked diffusion orchestrator
│   ├── data/
│   │   └── pipeline.py       # Complete data processing pipeline
│   ├── training/
│   │   ├── trainer.py        # Enhanced curriculum trainer
│   │   ├── scheduler.py      # Curriculum scheduling logic
│   │   ├── metrics.py        # Training metrics tracking
│   │   └── format_datasets.py # Stage-specific dataset formats
│   └── evaluation/
│       ├── generate.py       # Text generation engine
│       ├── metrics.py        # Evaluation metrics (style, coherence)
│       └── analysis.py       # Results analysis tools
├── scripts/                  # Entry points - standalone
│   ├── prepare_data.py       # Data preparation script
│   ├── train.py              # Training script with testing modes
│   ├── generate.py           # Text generation script
│   └── evaluate.py           # Model evaluation script
├── notebooks/                # Optional analysis tools
│   ├── data_exploration.ipynb
│   ├── curriculum_analysis.ipynb
│   └── results_analysis.ipynb
└── outputs/
    ├── checkpoints/
    ├── logs/
    ├── tensorboard/
    └── samples/
```

## Project Workflows

### **Development Workflow** (One-time Setup)
Prepare project foundation and architecture

### **Training Workflow** (Iterative Process)
Execute curriculum learning with monitoring

### **Inference Workflow** (Ongoing Usage)
Generate text and analyze results

## Configuration System

The project uses a comprehensive configuration system with cascading overrides:

### **Core Configuration Files**

**`config/__init__.py`** - **Unified Config Manager**
- **Purpose**: Single entry point for all configurations with override system
- **Features**: Command line → config file → programmatic overrides
- **Usage**: `from config import ProjectConfig; cfg = ProjectConfig.default()`

**`config/model_config.py`** - **Architecture Parameters**
- **Purpose**: 125+ configurable model parameters (d_model, layers, vocab, RoPE, etc.)
- **Presets**: `tiny_125m()`, `memory_efficient_125m()`, `experimental_125m()`
- **Features**: Parameter validation, memory estimation, comparable configs

**`config/training_config.py`** - **Training Hyperparameters**
- **Purpose**: Learning rates, batch sizes, hardware limits, memory optimization
- **Features**: Automatic VRAM estimation, gradient accumulation, mixed precision
- **Presets**: `default()`, `memory_efficient()`, `fast_debug()`

**`config/curriculum_config.py`** - **Learning Schedule**
- **Purpose**: 3-stage curriculum with masking rates, data selection, format types
- **Features**: Stage transitions, pseudo-data generation, difficulty thresholds
- **Presets**: `three_stage()`, `single_stage()`, `fast_debug()`, `research_config()`

**`config/generation_config.py`** - **Text Generation**
- **Purpose**: Temperature, top-k/top-p, diffusion steps, confidence scheduling
- **Features**: Multiple sampling strategies, style control, vocab level adaptation
- **Presets**: `default()`, `creative()`, `conservative()`, `fast()`

**`config/pipeline_config.py`** - **Data Processing**
- **Purpose**: All text processing parameters (difficulty scoring, clustering, etc.)
- **Features**: Lexical analysis, syntactic complexity, thematic centrality
- **Presets**: `default()`, `fast_processing()`, `high_quality()`

**`config/evaluation_config.py`** - **Evaluation & Benchmarking**
- **Purpose**: Evaluation metrics, quality thresholds, performance benchmarks
- **Features**: Perplexity calculation, style analysis, coherence metrics
- **Presets**: `default()`, `fast()`, `comprehensive()`, `memory_efficient()`

### **Configuration Usage Examples**

```python
# Default configuration
config = ProjectConfig.default()

# Override with dot notation
config = config.override(**{
    "model.d_model": 512,
    "training.batch_size": 16,
    "curriculum.stages[0].epochs": 25
})

# Load from config file
config = config.override_from_file("experiment.py")

# Override from command line
config = config.override_from_args(args)

# Quick presets
debug_config = ProjectConfig.debug()
research_config = ProjectConfig.comprehensive()
```

## File Descriptions (Execution Order)

### Development Workflow

#### Phase 1: Configuration Setup

**`config/__init__.py`** - **Unified Config Manager**
- **Purpose**: Single entry point for all configurations with override cascading
- **Usage**: `from config import ProjectConfig; cfg = ProjectConfig.default()`
- **Without it**: Manual imports, potential conflicts between config files

**`config/model_config.py`** - **Architecture Parameters**
- **Purpose**: Model size, layers, vocab, sequence length (125+ parameters)
- **Usage**: Defines transformer architecture with validation and presets
- **Without it**: Hardcoded values scattered throughout code

**`config/curriculum_config.py`** - **Learning Schedule**
- **Purpose**: Masking rates per stage (75-90%, 40-60%, 10-30%)
- **Usage**: Defines 3-stage curriculum progression with format types
- **Without it**: Random training order, no structured learning

**`config/training_config.py`** - **Training Hyperparameters**
- **Purpose**: Learning rate, batch size, optimizer, hardware limits
- **Usage**: Controls training loop behavior with memory estimation
- **Without it**: No systematic hyperparameter management

**`config/generation_config.py`** - **Text Generation Parameters**
- **Purpose**: Temperature, sampling, diffusion steps, confidence scheduling
- **Usage**: Controls text generation quality and style
- **Without it**: Default generation behavior, no fine-tuning control

**`config/pipeline_config.py`** - **Data Processing Parameters**
- **Purpose**: Text processing, difficulty scoring, clustering parameters
- **Usage**: Configures entire data pipeline processing
- **Without it**: Hardcoded processing parameters, no pipeline flexibility

**`config/evaluation_config.py`** - **Evaluation Parameters**
- **Purpose**: Metrics, thresholds, benchmarking parameters
- **Usage**: Controls evaluation quality and scope
- **Without it**: Basic evaluation only, no comprehensive analysis

#### Phase 2: Data Preparation

**`scripts/prepare_data.py`** - **Data Preparation Script**
- **Purpose**: Run data pipeline, create curriculum
- **Usage**: `python prepare_data.py --book path/to/book.txt`
- **Without it**: Manual data preparation

**`src/data/pipeline.py`** - **Complete Data Processing**
- **Purpose**: Text → sentences → difficulty scores → curriculum → dataset
- **Usage**: Single class handling entire data flow with config-driven processing
- **Without it**: Manual data processing, no curriculum construction

#### Phase 3: Model Architecture

**`src/model/attention.py`** - **Multi-Head Attention**
- **Purpose**: Bidirectional attention mechanism (no causal masking) with RoPE
- **Usage**: Core transformer component with configurable parameters
- **Without it**: No transformer architecture possible

**`src/model/transformer.py`** - **Transformer Blocks**
- **Purpose**: Layer norm, attention, feedforward layers with SwiGLU
- **Usage**: Stacks attention blocks with configurable architecture
- **Without it**: No neural network backbone

**`src/model/diffusion.py`** - **Masked Diffusion Logic**
- **Purpose**: Masking/unmasking, denoising objective, generation
- **Usage**: Orchestrates training and inference with configurable strategies
- **Without it**: Just a regular language model, not diffusion

### Training Workflow

#### Phase 4: Training

**`src/training/format_datasets.py`** - **Format-Aware Dataset Classes**
- **Purpose**: Stage-specific data formatting (sentences, pairs, paragraphs)
- **Usage**: Creates different input formats for each curriculum stage
- **Without it**: All stages would use identical format, missing structural learning

**`scripts/train.py`** - **Training Script**
- **Purpose**: Execute training loop with curriculum and config system
- **Usage**: `python train.py` or `python train.py --debug`
- **Without it**: No command-line training interface

**`src/training/trainer.py`** - **Enhanced Training Loop**
- **Purpose**: Curriculum-aware training with dynamic adaptation
- **Usage**: Main training orchestrator with config-driven behavior
- **Without it**: No systematic training process

**`src/training/scheduler.py`** - **Curriculum Scheduling**
- **Purpose**: Stage transitions, masking rate changes, adaptive scheduling
- **Usage**: Controls progression through 3 stages with config parameters
- **Without it**: Static training, no curriculum learning

**`src/training/metrics.py`** - **Training Metrics**
- **Purpose**: Loss tracking, perplexity calculation, progress monitoring
- **Usage**: Real-time training monitoring with configurable windows
- **Without it**: No training progress visibility

### Inference Workflow

#### Phase 5: Generation & Evaluation

**`scripts/generate.py`** - **Generation Script**
- **Purpose**: Generate text from trained checkpoint with vocab levels
- **Usage**: `python generate.py --prompt "The origin of" --vocab-level 5`
- **Without it**: No easy text generation interface

**`src/evaluation/generate.py`** - **Text Generation Engine**
- **Purpose**: Inference pipeline, sampling strategies, confidence control
- **Usage**: Generate text with configurable parameters
- **Without it**: No way to test model output

**`scripts/evaluate.py`** - **Evaluation Script**
- **Purpose**: Run comprehensive evaluation suite
- **Usage**: `python evaluate.py --checkpoint model.pt --data-dir data/processed`
- **Without it**: Manual evaluation process

**`src/evaluation/metrics.py`** - **Evaluation Metrics**
- **Purpose**: Style similarity, coherence analysis, quality assessment
- **Usage**: Comprehensive model evaluation with configurable thresholds
- **Without it**: No systematic performance measurement

**`src/evaluation/analysis.py`** - **Results Analysis**
- **Purpose**: Compare against baselines, visualizations, detailed analysis
- **Usage**: Comprehensive model evaluation and comparison
- **Without it**: No comparative analysis

#### Phase 6: Optional Analysis (LATER STAGE - NOT NEEDED)

**`notebooks/`** - **Interactive Analysis Tools**
- **Purpose**: Data exploration, curriculum analysis, results visualization
- **Usage**: Jupyter notebooks for deeper investigation
- **Without it**: Limited insight into model behavior and training dynamics

## Quick Start Guide

### 1. **Setup Environment**
```bash
# Install dependencies
pip install torch transformers sentence-transformers scikit-learn nltk spacy textstat

# Download spaCy model
python -m spacy download en_core_web_sm

# Test installation
python scripts/train.py --test              # Ultra-fast test (10s)
python scripts/train.py --test-integration  # Integration test (30s)
```

### 2. **Prepare Data**
```bash
# Clean previous runs
./clean.sh quick

# Prepare data from any book
python scripts/prepare_data.py --book data/raw/frankenstein.txt --output data/processed
```

### 3. **Train Model**
```bash
# Debug mode (fast, 1 epoch per stage)
python scripts/train.py --debug --data-dir data/processed

# Full training (3-stage curriculum)
python scripts/train.py --data-dir data/processed

# Memory-efficient training
python scripts/train.py --config memory_efficient --data-dir data/processed
```

### 4. **Generate Text**
```bash
# Generate from trained model
python scripts/generate.py --checkpoint outputs/best_model.pt --prompt "The origin of"

# Interactive generation
python scripts/generate.py --checkpoint outputs/best_model.pt --interactive

# Generate with specific vocab level
python scripts/generate.py --checkpoint outputs/best_model.pt --vocab-level 3 --prompt "Science"
```

### 5. **Evaluate Model**
```bash
# Full evaluation
python scripts/evaluate.py --checkpoint outputs/best_model.pt --data-dir data/processed

# Fast evaluation
python scripts/evaluate.py --checkpoint outputs/best_model.pt --data-dir data/processed --eval-batch-size 4 --eval-samples 50
```

## Testing Options

| Command | Duration | Purpose |
|---------|----------|---------|
| `--test` | 10s | Verify model architecture works |
| `--test-integration` | 30s | Test full training pipeline |
| `--debug` | Fast | Train with 1 epoch per stage |

## Potential Challenges & Mitigations

### Memory Constraints
- **Issue**: 8GB VRAM limitation
- **Solution**: Gradient checkpointing, smaller batch sizes, mixed precision training, memory-efficient config

### Overfitting Risk
- **Issue**: Single book training data
- **Solution**: Strong regularization, curriculum learning, early stopping on validation set

### Mode Collapse
- **Issue**: Repetitive generation
- **Solution**: Diverse masking strategies, temperature sampling, nucleus sampling

### Training Instability
- **Issue**: Curriculum transitions
- **Solution**: Gradual transitions, learning rate annealing, careful initialization

## Success Metrics

1. **Technical Success**: Model converges and generates coherent text
2. **Style Success**: Generated text recognizably matches source book style
3. **Educational Success**: Deep understanding of curriculum learning and diffusion models
4. **Fun Success**: Enjoy the process and share interesting results

## Resources & References

### Key Papers
- "Diffusion Beats Autoregressive in Data-Constrained Settings" (2025)
- "Empirical Use of Masked Diffusion Models for Text Generation" (2025)
- "Generative Stylography: Curriculum Learning Framework" (2025)

### Implementation References
- Hugging Face Transformers for model architecture
- PyTorch for deep learning framework
- Sentence Transformers for text embeddings
- spaCy and NLTK for text processing

---

*This project represents a practical exploration of cutting-edge research in a fun, educational context. The comprehensive configuration system and cleanup tools make it easy to experiment with different approaches while maintaining code quality and reproducibility.*