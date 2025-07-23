# Tiny Text Diffusion Model: Single-Book Style Generator

## Project Overview

A comprehensive exploration of training a small masked diffusion language model on any single book using consumer hardware. This project implements cutting-edge curriculum learning strategies to maximize performance in data-constrained settings, based on 2025 state-of-the-art research.

### Key Research Insights
- **Diffusion beats autoregressive in data-constrained settings**: When data is limited but compute is available, masked diffusion models significantly outperform AR models
- **Critical compute threshold**: Diffusion becomes favorable when compute exceeds `Ccrit(U) = 2.12 × 10^15 · U^2.174` FLOPs
- **Superior data reuse**: Diffusion models can benefit from ~500 epochs vs ~15 for AR models
- **Curriculum learning essential**: 3-stage progression maximizes small model capabilities

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
- **SwiGLU Activation**: Improved convergence over standard GELU/ReLU

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

#### Multi-Dimensional Difficulty Scoring
```python
# Comprehensive difficulty assessment:
difficulty_scores = {
    'lexical_rarity': calculate_idf_scores(sentences, general_corpus),
    'syntactic_complexity': calculate_readability_scores(sentences),
    'argument_structure': parse_argumentative_roles(sentences),  # Optional for narrative
    'thematic_centrality': calculate_cluster_centrality(sentence_embeddings)
}
```

#### Curriculum Construction
- **Easy Examples**: High centrality + low complexity + common vocabulary
- **Medium Examples**: Moderate complexity + clear argumentative structure
- **Hard Examples**: All data including outliers and complex arguments

## File Structure

```
tiny-diffusion/
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py           # Simple unified config dataclass
│   ├── model.py              # Architecture + generation parameters  
│   └── curriculum.py         # 3-stage curriculum (core research innovation)
├── src/
│   ├── model.py              # Complete model: Attention + Transformer + Diffusion
│   ├── data.py               # Data pipeline + curriculum + difficulty scoring
│   ├── trainer.py            # Training loop + scheduling + metrics
│   └── evaluation.py         # Generation + style analysis + benchmarking
├── scripts/
│   ├── train.py              # Main entry point (includes data preparation)
│   └── generate.py           # Standalone text generation
├── data/
│   ├── raw/
│   │   └── [your_book].txt
│   ├── processed/
│   │   ├── segments.pkl
│   │   ├── curriculum_splits.pkl
│   │   ├── vocab_curriculum.pkl
│   │   └── dynamic_scorer.pkl
│   └── tokenizer/
│       └── compressed_tokenizer.json
└── outputs/
    ├── checkpoints/
    ├── logs/
    ├── tensorboard/
    └── samples/
```

## Implementation Components

### Configuration System

**`config/__init__.py`** - **Unified Config Manager**
- **Purpose**: Single dataclass-based configuration with override cascading
- **Features**: Command line → config file → programmatic overrides
- **Usage**: `from config import ProjectConfig; cfg = ProjectConfig.default()`

**`config/model.py`** - **Architecture + Generation Parameters**
- **Purpose**: Model size, layers, vocab, sequence length + generation settings
- **Features**: Parameter validation, memory estimation, sampling strategies
- **Presets**: `tiny_125m()`, `memory_efficient()`, `creative_generation()`

**`config/curriculum.py`** - **3-Stage Learning Schedule**
- **Purpose**: Masking rates per stage, data selection criteria, format specifications
- **Features**: Stage transitions, difficulty thresholds, pseudo-data generation
- **Usage**: Defines the core research innovation of curriculum learning

### Core Implementation

**`src/model.py`** - **Complete Model Architecture**
- **Purpose**: All model components in one coherent file
- **Components**:
  - `MultiHeadAttention`: Bidirectional attention with RoPE
  - `TransformerBlock`: Layer norm, attention, SwiGLU feedforward
  - `MaskedDiffusionLM`: Full model orchestrating training and inference
- **Features**: Memory-efficient implementation, gradient checkpointing

**`src/data.py`** - **Data Pipeline + Curriculum Construction**
- **Purpose**: Complete data processing from raw text to curriculum-ready datasets
- **Features**:
  - Text segmentation and preprocessing
  - Multi-dimensional difficulty scoring
  - Cluster analysis for centrality calculation
  - Stage-specific dataset formatting
  - Compressed tokenizer creation

**`src/trainer.py`** - **Training Orchestrator**
- **Purpose**: End-to-end training with curriculum progression
- **Features**:
  - 3-stage curriculum execution
  - Dynamic masking rate adjustment
  - Memory-efficient training loops
  - Real-time metrics tracking
  - Stage transition management

**`src/evaluation.py`** - **Generation + Analysis**
- **Purpose**: Text generation and comprehensive evaluation
- **Components**:
  - `TextGenerator`: Inference pipeline with sampling strategies
  - `StyleAnalyzer`: Stylometric analysis and similarity metrics
  - `Benchmarker`: Performance comparison tools
- **Features**: Interactive generation, style fidelity assessment, coherence analysis

### Entry Points

**`scripts/train.py`** - **Main Training Script**
- **Purpose**: Unified entry point for complete training workflow
- **Features**:
  - Integrated data preparation
  - 3-stage curriculum execution
  - Built-in evaluation
  - Debug and test modes
- **Usage**: `python scripts/train.py --book frankenstein.txt --debug`

**`scripts/generate.py`** - **Standalone Generation**
- **Purpose**: Interactive text generation from trained models
- **Features**: Multiple sampling strategies, prompt continuation, style control
- **Usage**: `python scripts/generate.py --checkpoint best_model.pt --prompt "Science"`

## Training Configuration

```python
# Hyperparameters (based on 2025 research)
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

### 2. **Prepare and Train**
```bash
# Train with integrated data preparation (debug mode)
python scripts/train.py --book data/raw/frankenstein.txt --debug

# Full training (3-stage curriculum)
python scripts/train.py --book data/raw/frankenstein.txt

# Memory-efficient training
python scripts/train.py --book data/raw/frankenstein.txt --memory-efficient
```

### 3. **Generate Text**
```bash
# Generate from trained model
python scripts/generate.py --checkpoint outputs/best_model.pt --prompt "The origin of"

# Interactive generation
python scripts/generate.py --checkpoint outputs/best_model.pt --interactive

# Generate with specific style parameters
python scripts/generate.py --checkpoint outputs/best_model.pt --temperature 0.8 --prompt "Science"
```

## Configuration Usage Examples

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

# Quick presets
debug_config = ProjectConfig.debug()
memory_config = ProjectConfig.memory_efficient()
```

## Expected Outcomes

### Performance Targets
- **Perplexity**: <15 on held-out text (baseline AR model likely ~20)
- **Style Fidelity**: Generated text matches author's sentence length distribution
- **Coherence**: Logical flow in generated passages
- **Novelty**: Generate new content, not memorized passages

### Learning Trajectory
Based on research, expect:
- **Stage I**: Rapid initial learning, plateau around epoch 30
- **Stage II**: Steady improvement in logical structure
- **Stage III**: Gradual refinement, potential for 200+ beneficial epochs

## Testing Options

| Command | Duration | Purpose |
|---------|----------|---------|
| `--test` | 10s | Verify model architecture works |
| `--test-integration` | 30s | Test full training pipeline |
| `--debug` | Fast | Train with 1 epoch per stage |

## Potential Challenges & Mitigations

### Memory Constraints
- **Issue**: 8GB VRAM limitation
- **Solution**: Gradient checkpointing, smaller batch sizes, mixed precision training

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
- "Tiny GPT Model Training Research" (2025)

### Implementation References
- Hugging Face Transformers for model architecture
- PyTorch for deep learning framework
- Sentence Transformers for text embeddings
- spaCy and NLTK for text processing

## Project Philosophy

This project represents a practical exploration of cutting-edge research in a fun, educational context. The simplified architecture focuses on:

- **Research Innovation**: Implementing 2025 state-of-the-art curriculum learning
- **Educational Value**: Clear, understandable code structure
- **Practical Results**: Working text generation that captures book styles
- **Extensibility**: Easy to modify and experiment with

The goal is to demonstrate that small, well-trained diffusion models can achieve remarkable results when guided by intelligent curriculum design, proving that innovation beats brute force scaling.

---

*This project bridges cutting-edge academic research with hands-on implementation, making advanced AI techniques accessible and educational while producing genuinely useful results.*