# Hardware Compatibility & Setup Guide

## System Verification ✅

Your hardware is **excellent** for this project:
- **RTX 3070 Ti (8GB)**: Perfect for ~125M parameter model with gradient checkpointing
- **64GB RAM**: More than sufficient for data preprocessing and caching
- **Intel i7-12700H**: Strong CPU for data pipeline operations

## CUDA Setup

Check your CUDA version:
```bash
nvidia-smi
nvcc --version
```

Your RTX 3070 Ti supports CUDA 11.x and 12.x. The requirements.txt uses **CUDA 12.1** (most stable).

## Installation Steps

### 1. Create Environment
```bash
# Using conda (recommended)
conda create -n tiny-diffusion python=3.10
conda activate tiny-diffusion

# Or using venv
python -m venv tiny-diffusion
# Windows: tiny-diffusion\Scripts\activate
# Linux/Mac: source tiny-diffusion/bin/activate
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Verify GPU Access
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

### 4. Download SpaCy Model (for text processing)
```bash
python -m spacy download en_core_web_sm
```

## Memory Optimization Settings

Add to your training scripts:
```python
# Memory efficient settings for 8GB VRAM
BATCH_SIZE = 32  # Start here, can go up to 64
GRADIENT_ACCUMULATION = 2
MIXED_PRECISION = True
GRADIENT_CHECKPOINTING = True
```

## Estimated Resource Usage

**Training Stage 1 (Foundation):**
- Model: ~125M params = 500MB
- Batch: 32 × 512 tokens = 400MB  
- Gradients: ~500MB
- **Total: ~2GB VRAM** ✅

**Training Stage 3 (Refinement):**
- With gradient checkpointing: ~4-5GB VRAM ✅
- Without: ~7-8GB VRAM (tight but doable)

## Troubleshooting

**CUDA Out of Memory:**
- Reduce `BATCH_SIZE` to 16 or 8
- Enable gradient checkpointing
- Use `torch.cuda.empty_cache()` between stages

**Slow Data Loading:**
- Use `num_workers=4` (based on your 14 cores)
- Enable `pin_memory=True`

**Installation Issues:**
- If PyTorch CUDA fails: Try `pip install torch --force-reinstall`
- If deepspeed fails: Skip it, not essential for this project size
