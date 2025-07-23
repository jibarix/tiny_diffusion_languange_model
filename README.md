# Tiny Text Diffusion Model

A research implementation of masked diffusion language models with curriculum learning, designed for training on single books using consumer hardware.

## Quick Start

```bash
# Install dependencies
pip install torch transformers sentence-transformers scikit-learn nltk spacy textstat

# Download spaCy model
python -m spacy download en_core_web_sm

# Test installation
python scripts/train.py --test              # Ultra-fast test (10s)
python scripts/train.py --test-integration  # Integration test (30s)
```

## Usage

### 1. Prepare Data
```bash
python scripts/prepare_data.py --book path/to/your_book.txt --output data/processed
```

### 2. Train Model
```bash
# Full training (3-stage curriculum)
python scripts/train.py --config config.yaml --data-dir data/processed

# Debug mode (fast, 1 epoch per stage)
python scripts/train.py --debug --data-dir data/processed
```

### 3. Generate Text
```bash
python scripts/generate.py --checkpoint outputs/best_model.pt --prompt "The origin of"
```

### 4. Evaluate
```bash
python scripts/evaluate.py --checkpoint outputs/best_model.pt --data-dir data/processed
```

## Testing Options

| Command | Duration | Purpose |
|---------|----------|---------|
| `--test` | 10s | Verify model architecture works |
| `--test-integration` | 30s | Test full training pipeline |
| `--debug` | Fast | Train with 1 epoch per stage |

## Key Features

- **Masked Diffusion**: Bidirectional text generation via iterative demasking
- **3-Stage Curriculum**: Foundation → Structural → Refinement learning
- **Multi-dimensional Difficulty**: Lexical, syntactic, thematic, and argumentative complexity
- **Consumer Hardware**: Optimized for 8GB VRAM (RTX 3070 Ti)
- **Compressed Tokenizer**: Reduces vocab from 50k to 25k tokens
- **Dynamic Adaptation**: Curriculum adjusts based on training performance

## Architecture

- **Size**: ~125M parameters (12 layers, 768 hidden, deeper-not-wider design)
- **Attention**: Bidirectional with RoPE positional encoding
- **Curriculum**: Easy (75-90% masking) → Medium (40-60%) → Hard (10-30%)
- **Hardware**: Trains on single RTX 3070 Ti with gradient checkpointing

## Project Structure

```
tiny-diffusion/
├── config/           # Model, training, curriculum configs
├── src/
│   ├── model/        # Diffusion transformer architecture
│   ├── data/         # Text processing and curriculum pipeline
│   ├── training/     # Enhanced curriculum trainer
│   └── evaluation/   # Generation and metrics
├── scripts/          # Entry points for training/evaluation
└── outputs/          # Checkpoints, logs, generated samples
```

## Research Basis

Based on recent findings that **diffusion beats autoregressive in data-constrained settings** when compute exceeds the critical threshold `Ccrit(U) = 2.12 × 10^15 · U^2.174` FLOPs. Implements curriculum learning strategies for small language models as detailed in current literature.

## Hardware Requirements

- **Minimum**: RTX 3070 Ti (8GB VRAM), 32GB RAM
- **Recommended**: RTX 4080+ (16GB VRAM), 64GB RAM
- **CPU**: Intel i7-12700H or equivalent
- **Storage**: 10GB for model + data

## License

Research use only. Based on academic papers in masked diffusion and curriculum learning.