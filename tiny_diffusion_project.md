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
- **Vocab Size**: ~25k tokens (compressed from 50k GPT-2 vocab using Darwin corpus frequency analysis)
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
- [ ] Load and clean any book text
- [ ] Implement tokenizer compression based on book vocabulary
- [ ] Create difficulty scoring system
- [ ] Generate curriculum schedules for all three stages

### Phase 2: Model Implementation
- [ ] Implement masked diffusion transformer in PyTorch
- [ ] Create training loop with curriculum scheduling
- [ ] Implement masking strategies (uniform random masking)
- [ ] Add logging and checkpointing

### Phase 3: Training
- [ ] Stage I training (50 epochs)
- [ ] Stage II training (100 epochs)
- [ ] Stage III training (150+ epochs)
- [ ] Monitor for convergence and adjust as needed

### Phase 4: Evaluation
- [ ] Perplexity on held-out text
- [ ] Style similarity analysis (sentence length distribution, vocabulary usage)
- [ ] Human evaluation of generated samples
- [ ] Comparison with baseline autoregressive model

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
tiny-diffusion-darwin/
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py           # Unified ProjectConfig namespace
│   ├── model_config.py
│   ├── training_config.py    # Imports model + curriculum configs
│   └── curriculum_config.py
├── data/
│   ├── raw/
│   │   └── [your_book].txt
│   ├── processed/
│   │   ├── sentences.pkl
│   │   ├── difficulty_scores.pkl
│   │   └── curriculum_data.pkl
│   └── tokenizer/
│       └── book_tokenizer.json
├── src/
│   ├── model/
│   │   ├── attention.py      # Used by transformer.py
│   │   ├── transformer.py    # Uses attention.py
│   │   └── diffusion.py      # Orchestrates transformer
│   ├── data/
│   │   └── pipeline.py       # Unified: preprocess → curriculum → dataset
│   ├── training/
│   │   ├── trainer.py
│   │   ├── scheduler.py
│   │   └── metrics.py        # Training-specific metrics
│   └── evaluation/
│       ├── generate.py
│       ├── metrics.py        # Evaluation-specific metrics  
│       └── analysis.py
├── scripts/                  # Entry points - standalone
│   ├── prepare_data.py
│   ├── train.py
│   ├── generate.py
│   └── evaluate.py
├── notebooks/                # Independent analysis tools
│   ├── data_exploration.ipynb
│   ├── curriculum_analysis.ipynb
│   └── results_analysis.ipynb
└── outputs/
    ├── checkpoints/
    ├── logs/
    └── samples/
```

## Project Workflows

### **Development Workflow** (One-time Setup)
Prepare project foundation and architecture

### **Training Workflow** (Iterative Process)
Execute curriculum learning with monitoring

### **Inference Workflow** (Ongoing Usage)
Generate text and analyze results

## File Descriptions (Execution Order)

### Development Workflow

#### Phase 1: Configuration Setup

**`config/__init__.py`** - **Unified Config Manager**
- **Purpose**: Single entry point for all configurations
- **Usage**: `from config import ProjectConfig; cfg = ProjectConfig()`
- **Without it**: Manual imports, potential conflicts between config files

**`config/model_config.py`** - **Architecture Parameters**
- **Purpose**: Model size, layers, vocab, sequence length
- **Usage**: Defines transformer architecture (125M params, 12 layers, etc.)
- **Without it**: Hardcoded values scattered throughout code

**`config/curriculum_config.py`** - **Learning Schedule**
- **Purpose**: Masking rates per stage (75-90%, 40-60%, 10-30%)
- **Usage**: Defines 3-stage curriculum progression
- **Without it**: Random training order, no structured learning

**`config/training_config.py`** - **Training Hyperparameters**
- **Purpose**: Learning rate, batch size, optimizer settings, imports curriculum schedules
- **Usage**: Controls training loop behavior
- **Without it**: No systematic hyperparameter management

#### Phase 2: Data Preparation

**`scripts/prepare_data.py`** - **Data Preparation Script**
- **Purpose**: Run data pipeline, create curriculum
- **Usage**: `python prepare_data.py --book path/to/book.txt`
- **Without it**: Manual data preparation

**`src/data/pipeline.py`** - **Complete Data Processing**
- **Purpose**: Text → sentences → difficulty scores → curriculum → dataset
- **Usage**: Single class handling entire data flow
- **Without it**: Manual data processing, no curriculum construction

#### Phase 3: Model Architecture

**`src/model/attention.py`** - **Multi-Head Attention**
- **Purpose**: Bidirectional attention mechanism (no causal masking)
- **Usage**: Core transformer component, imported by transformer.py
- **Without it**: No transformer architecture possible

**`src/model/transformer.py`** - **Transformer Blocks**
- **Purpose**: Layer norm, attention, feedforward layers
- **Usage**: Stacks attention blocks, imported by diffusion.py
- **Without it**: No neural network backbone

**`src/model/diffusion.py`** - **Masked Diffusion Logic**
- **Purpose**: Masking/unmasking, denoising objective
- **Usage**: Orchestrates training and inference
- **Without it**: Just a regular language model, not diffusion

### Training Workflow

#### Phase 4: Training

**`scripts/train.py`** - **Training Script**
- **Purpose**: Execute training loop with curriculum
- **Usage**: `python train.py --config config.yaml`
- **Without it**: No command-line training interface

**`src/training/trainer.py`** - **Training Loop**
- **Purpose**: Epoch management, loss computation, checkpointing
- **Usage**: Main training orchestrator
- **Without it**: No systematic training process

**`src/training/scheduler.py`** - **Curriculum Scheduling**
- **Purpose**: Stage transitions, masking rate changes
- **Usage**: Controls progression through 3 stages
- **Without it**: Static training, no curriculum learning

**`src/training/metrics.py`** - **Training Metrics**
- **Purpose**: Loss tracking, perplexity calculation during training
- **Usage**: Real-time training monitoring
- **Without it**: No training progress visibility

### Inference Workflow

#### Phase 5: Generation & Evaluation

**`scripts/generate.py`** - **Generation Script**
- **Purpose**: Generate text from trained checkpoint
- **Usage**: `python generate.py --prompt "The origin of"`
- **Without it**: No easy text generation

**`src/evaluation/generate.py`** - **Text Generation**
- **Purpose**: Inference pipeline, sampling strategies
- **Usage**: Generate text from trained model
- **Without it**: No way to test model output

**`scripts/evaluate.py`** - **Evaluation Script**
- **Purpose**: Run full evaluation suite
- **Usage**: `python evaluate.py --checkpoint model.pt`
- **Without it**: Manual evaluation process

**`src/evaluation/metrics.py`** - **Evaluation Metrics**
- **Purpose**: Style similarity, coherence analysis, final assessment
- **Usage**: Model quality evaluation
- **Without it**: No systematic performance measurement

**`src/evaluation/analysis.py`** - **Results Analysis**
- **Purpose**: Compare against baselines, visualizations
- **Usage**: Comprehensive model evaluation
- **Without it**: No comparative analysis

#### Phase 6: Optional Analysis

**`notebooks/`** - **Interactive Analysis Tools**
- **Purpose**: Data exploration, curriculum analysis, results visualization
- **Usage**: Jupyter notebooks for deeper investigation
- **Without it**: Limited insight into model behavior and training dynamics

### Phase 1: Configuration Setup

**`config/__init__.py`** - **Unified Config Manager**
- **Purpose**: Single entry point for all configurations
- **Usage**: `from config import ProjectConfig; cfg = ProjectConfig()`
- **Without it**: Manual imports, potential conflicts between config files

**`config/model_config.py`** - **Architecture Parameters**
- **Purpose**: Model size, layers, vocab, sequence length
- **Usage**: Defines transformer architecture (125M params, 12 layers, etc.)
- **Without it**: Hardcoded values scattered throughout code

**`config/curriculum_config.py`** - **Learning Schedule**
- **Purpose**: Masking rates per stage (75-90%, 40-60%, 10-30%)
- **Usage**: Defines 3-stage curriculum progression
- **Without it**: Random training order, no structured learning

**`config/training_config.py`** - **Training Hyperparameters**
- **Purpose**: Learning rate, batch size, optimizer settings, imports curriculum schedules
- **Usage**: Controls training loop behavior
- **Without it**: No systematic hyperparameter management

### Phase 2: Data Preparation

**`scripts/prepare_data.py`** - **Data Preparation Script**
- **Purpose**: Run data pipeline, create curriculum
- **Usage**: `python prepare_data.py --book path/to/book.txt`
- **Without it**: Manual data preparation

**`src/data/pipeline.py`** - **Complete Data Processing**
- **Purpose**: Text → sentences → difficulty scores → curriculum → dataset
- **Usage**: Single class handling entire data flow
- **Without it**: Manual data processing, no curriculum construction

### Phase 3: Model Architecture

**`src/model/attention.py`** - **Multi-Head Attention**
- **Purpose**: Bidirectional attention mechanism (no causal masking)
- **Usage**: Core transformer component, imported by transformer.py
- **Without it**: No transformer architecture possible

**`src/model/transformer.py`** - **Transformer Blocks**
- **Purpose**: Layer norm, attention, feedforward layers
- **Usage**: Stacks attention blocks, imported by diffusion.py
- **Without it**: No neural network backbone

**`src/model/diffusion.py`** - **Masked Diffusion Logic**
- **Purpose**: Masking/unmasking, denoising objective
- **Usage**: Orchestrates training and inference
- **Without it**: Just a regular language model, not diffusion

### Phase 4: Training

**`scripts/train.py`** - **Training Script**
- **Purpose**: Execute training loop with curriculum
- **Usage**: `python train.py --config config.yaml`
- **Without it**: No command-line training interface

**`src/training/trainer.py`** - **Training Loop**
- **Purpose**: Epoch management, loss computation, checkpointing
- **Usage**: Main training orchestrator
- **Without it**: No systematic training process

**`src/training/scheduler.py`** - **Curriculum Scheduling**
- **Purpose**: Stage transitions, masking rate changes
- **Usage**: Controls progression through 3 stages
- **Without it**: Static training, no curriculum learning

**`src/training/metrics.py`** - **Training Metrics**
- **Purpose**: Loss tracking, perplexity calculation during training
- **Usage**: Real-time training monitoring
- **Without it**: No training progress visibility

### Phase 5: Generation & Evaluation

**`scripts/generate.py`** - **Generation Script**
- **Purpose**: Generate text from trained checkpoint
- **Usage**: `python generate.py --prompt "The origin of"`
- **Without it**: No easy text generation

**`src/evaluation/generate.py`** - **Text Generation**
- **Purpose**: Inference pipeline, sampling strategies
- **Usage**: Generate text from trained model
- **Without it**: No way to test model output

**`scripts/evaluate.py`** - **Evaluation Script**
- **Purpose**: Run full evaluation suite
- **Usage**: `python evaluate.py --checkpoint model.pt`
- **Without it**: Manual evaluation process

**`src/evaluation/metrics.py`** - **Evaluation Metrics**
- **Purpose**: Style similarity, coherence analysis, final assessment
- **Usage**: Model quality evaluation
- **Without it**: No systematic performance measurement

**`src/evaluation/analysis.py`** - **Results Analysis**
- **Purpose**: Compare against baselines, visualizations
- **Usage**: Comprehensive model evaluation
- **Without it**: No comparative analysis

### Phase 6: Optional Analysis

**`notebooks/`** - **Interactive Analysis Tools**
- **Purpose**: Data exploration, curriculum analysis, results visualization
- **Usage**: Jupyter notebooks for deeper investigation
- **Without it**: Limited insight into model behavior and training dynamics

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
2. **Style Success**: Generated text recognizably "Darwin-esque"
3. **Educational Success**: Deep understanding of curriculum learning and diffusion models
4. **Fun Success**: Enjoy the process and share interesting results

## Timeline: 5-Week Project

- **Week 1**: Data preparation and curriculum design
- **Week 2**: Model implementation and testing
- **Week 3**: Stage I & II training
- **Week 4**: Stage III training and hyperparameter tuning
- **Week 5**: Evaluation, analysis, and documentation

## Resources & References

### Key Papers
- "Diffusion Beats Autoregressive in Data-Constrained Settings" (2025)
- "Empirical Use of Masked Diffusion Models for Text Generation" (2025)
- "Generative Stylography: Curriculum Learning Framework" (2025)

### Implementation References
- Hugging Face Transformers for model architecture
- PyTorch Lightning for training infrastructure
- Weights & Biases for experiment tracking

---

*This project represents a practical exploration of cutting-edge research in a fun, educational context. While the goal is enjoyment and learning, we're committed to following best practices and rigorous evaluation.*