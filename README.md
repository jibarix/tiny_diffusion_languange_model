# Tiny Text Diffusion Model

A curriculum learning framework for training text diffusion models on literary texts. This project implements a 3-stage training curriculum designed to learn stylistic writing patterns from single books.

## Quick Start

### Installation
```bash
# Clone and install dependencies
pip install -r requirements.txt

# Download required NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Test installation (10 seconds)
python scripts/train.py --test

# Test full pipeline (30 seconds) 
python scripts/train.py --test-integration
```

### Data Preparation

#### Supported Text Formats
- Plain text files (.txt) 
- UTF-8 encoding required
- Any literary work or coherent text corpus
- Minimum ~50KB text recommended for meaningful curriculum

#### Text Requirements
```bash
# Acceptable input files:
data/raw/frankenstein.txt        # Classic literature
data/raw/darwin_origin.txt       # Scientific text  
data/raw/custom_corpus.txt       # Any coherent text

# The system handles:
# - Automatic sentence segmentation
# - Vocabulary compression (reduces 50K+ words to ~25K tokens)
# - Multi-dimensional difficulty scoring
# - 3-stage curriculum construction
```

#### Automatic Processing Pipeline
The training script automatically handles all data preparation:

1. **Text Segmentation**: Sentences split using NLTK + spaCy
2. **Difficulty Scoring**: 
   - Lexical rarity (IDF-based vocabulary complexity)
   - Syntactic complexity (readability metrics)
   - Thematic centrality (sentence embedding clusters)
3. **Compressed Tokenizer**: Custom vocabulary from your text
4. **Curriculum Construction**: Easy→Medium→Hard progression

```bash
# No separate data prep needed - training handles everything:
python scripts/train.py --book data/raw/your_text.txt
```

#### Manual Data Preparation (Optional)
```bash
# If you want to inspect data preparation separately:
python scripts/prepare_data.py --book data/raw/frankenstein.txt --output data/processed/

# This creates:
# - data/processed/segments.pkl (scored text segments)
# - data/processed/curriculum_splits.pkl (3-stage curriculum)
# - data/processed/compressed_tokenizer.json (custom vocab)
# - data/processed/statistics.json (dataset stats)
```

### Training
```bash
# Quick debug training (1 epoch per stage)
python scripts/train.py --book data/raw/frankenstein.txt --debug

# Full training with 3-stage curriculum
python scripts/train.py --book data/raw/frankenstein.txt

# Memory-efficient training for limited VRAM
python scripts/train.py --book data/raw/frankenstein.txt --memory-efficient
```

### Text Generation
```bash
# Generate from trained model
python scripts/generate.py --checkpoint outputs/best_model.pt --prompt "The origin of"

# Interactive generation mode
python scripts/generate.py --checkpoint outputs/best_model.pt --interactive

# Generate with custom parameters
python scripts/generate.py --checkpoint outputs/best_model.pt --temperature 0.8 --prompt "Science"
```

## Training Methodology

### 3-Stage Curriculum Learning

The model uses progressive difficulty training across three stages:

1. **Foundational (Stage I)**: Short, simple sentences for basic language patterns
2. **Intermediate (Stage II)**: Medium complexity text for structural understanding  
3. **Advanced (Stage III)**: Full complexity text including long, complex passages

Each stage uses different masking strategies and difficulty metrics to gradually increase challenge.

### Difficulty Scoring

Text difficulty is measured using:
- **Sentence length**: Shorter sentences are easier
- **Vocabulary complexity**: Common words vs. rare terms
- **Syntactic complexity**: Simple vs. complex sentence structures
- **Coherence requirements**: Local vs. global consistency needs

## Configuration

### Default Training
```python
# Uses balanced settings for most hardware
config = ProjectConfig.default()
```

### Debug Mode
```python
# Fast training with 1 epoch per stage
config = ProjectConfig.debug()
```

### Memory-Efficient Mode
```python
# Reduced batch size and gradient checkpointing
config = ProjectConfig.memory_efficient()
```

### Custom Configuration
```python
config = ProjectConfig.default().override(**{
    "model.d_model": 512,
    "training.batch_size": 16,
    "curriculum.stages[0].epochs": 25
})
```

## Expected Performance

### Training Targets
- **Perplexity**: <15 on validation text (baseline autoregressive ~20)
- **Style Fidelity**: Generated text matches source author's patterns
- **Coherence**: Logical flow in generated passages
- **Training Time**: ~5-8 hours on consumer GPU for full curriculum

### Learning Progression
- **Stage I**: Rapid initial learning, plateau around epoch 30
- **Stage II**: Steady improvement in logical structure  
- **Stage III**: Gradual refinement, benefits from extended training

## Hardware Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (GTX 3070, RTX 4060 Ti, or better)
- **RAM**: 16GB system memory
- **Storage**: 2GB free space for model checkpoints

### Optimization Features
- Gradient checkpointing for memory efficiency
- Mixed precision training (FP16)
- Dynamic batch sizing based on available memory
- Automatic device detection (CUDA/MPS/CPU)

## Project Structure

```
├── scripts/
│   ├── train.py          # Main training script
│   ├── generate.py       # Text generation script
│   └── evaluate.py       # Model evaluation
├── src/
│   ├── model/           # Diffusion model architecture
│   ├── training/        # Training and curriculum logic
│   ├── data/           # Data processing pipeline
│   └── evaluation/     # Style analysis and metrics
├── data/
│   └── raw/            # Input text files
└── outputs/            # Model checkpoints and logs
```

## Usage Examples

### Basic Training Workflow
```bash
# 1. Test your setup
python scripts/train.py --test

# 2. Train on your book
python scripts/train.py --book data/raw/your_book.txt

# 3. Generate text
python scripts/generate.py --checkpoint outputs/best_model.pt --prompt "Your prompt"

# 4. Evaluate results
python scripts/evaluate.py --checkpoint outputs/best_model.pt
```

### Advanced Options
```bash
# Resume from checkpoint
python scripts/train.py --book data/raw/book.txt --resume outputs/checkpoint_stage_2.pt

# Custom output directory
python scripts/train.py --book data/raw/book.txt --output custom_experiment/

# Specific GPU device
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --book data/raw/book.txt
```

## Troubleshooting

### Common Issues

**Out of Memory Error**
```bash
# Use memory-efficient mode
python scripts/train.py --book data/raw/book.txt --memory-efficient

# Or reduce batch size manually
python scripts/train.py --book data/raw/book.txt --batch-size 4
```

**Training Instability**
- Lower learning rate: `--learning-rate 1e-5`
- Enable gradient clipping: `--gradient-clip 1.0`
- Use mixed precision: `--mixed-precision` (default)

**Poor Generation Quality**
- Train longer: Increase epochs in Stage III
- Adjust temperature: Try values between 0.6-1.0
- Check validation loss: Should be <3.0 for good results

### Performance Optimization
- Use `--compile` for 10-15% speedup (PyTorch 2.0+)
- Enable `--channels-last` memory format
- Set `OMP_NUM_THREADS=4` for CPU efficiency

## Evaluation Metrics

The training process tracks:
- **Loss**: Cross-entropy loss per stage
- **Perplexity**: Model uncertainty measure
- **Style Similarity**: Sentence length distribution matching
- **Coherence Score**: Local and global text consistency
- **Generation Quality**: Human-readable output assessment

## Research Background

This implementation is based on recent research showing diffusion models outperform autoregressive models in data-constrained settings. Key papers:
- "Diffusion Beats Autoregressive in Data-Constrained Settings" (2025)
- "Empirical Use of Masked Diffusion Models for Text Generation" (2025)
- "Generative Stylography: Curriculum Learning Framework" (2025)

## Contributing

To extend this project:
1. Add new difficulty metrics in `src/data/difficulty.py`
2. Implement new curriculum strategies in `src/training/curriculum.py`
3. Create custom evaluation metrics in `src/evaluation/metrics.py`

## License

This project is for educational and research purposes. Model outputs should be used responsibly and in accordance with the source text's copyright.