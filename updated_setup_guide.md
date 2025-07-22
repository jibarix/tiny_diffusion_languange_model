# Hardware Compatibility & Setup Guide

## System Verification ✅

Your hardware is **excellent** for this project:
- **RTX 3070 Ti (8GB)**: Perfect for ~125M parameter model with gradient checkpointing
- **64GB RAM**: More than sufficient for data preprocessing and caching
- **Intel i7-12700H**: Strong CPU for data pipeline operations

## Setting Up Conda and VS Code on Windows

### 1. Install Miniconda or Anaconda
- Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)
- During install, you may check **"Add Miniconda3 to my PATH"** (recommended for devs)
- If you leave it unchecked, use "Anaconda Prompt" for all commands, or add to PATH manually

### 2. Add Conda to PATH (if needed)
If you see a "`conda` not recognized" error, add these to your user/system PATH:
- `C:\Users\<yourname>\anaconda3`
- `C:\Users\<yourname>\anaconda3\Scripts`
- `C:\Users\<yourname>\anaconda3\Library\bin`

*Restart your computer or all open terminals after changing PATH.*

### 3. Open a Terminal
Use **Anaconda Prompt**, **Command Prompt**, or **PowerShell**. In **VS Code**, use the integrated terminal (PowerShell is fine).

### 4. (First time only) Initialize Conda for PowerShell
```powershell
conda init powershell
```
*Restart PowerShell after this, so that environment names show up in your prompt.*

### 5. Create and Activate Your Environment
```sh
conda create -n tiny-diffusion python=3.10
conda activate tiny-diffusion
```

### 6. Verify Python & Environment
```sh
python --version
```
*Should show Python 3.10.x*

```powershell
Get-Command python
```
*Should show a path like `C:\Users\<yourname>\anaconda3\envs\tiny-diffusion\python.exe`*

### 7. (Optional) Open VS Code from Activated Environment
In your project directory:
```sh
code .
```
*This ensures VS Code uses your conda environment for Python.*

## CUDA Setup and Dependencies

Check your CUDA version:
```bash
nvidia-smi
nvcc --version
```

Your RTX 3070 Ti supports CUDA 11.x and 12.x. We'll use **CUDA 12.1** (most stable).

### 8. Install PyTorch with CUDA First
**Important:** Install PyTorch with CUDA before other dependencies:

```sh
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 9. Install Remaining Dependencies
```sh
pip install -r requirements.txt
```

### 10. Downgrade NumPy for Compatibility
PyTorch with CUDA works best with **NumPy 1.x**:
```sh
pip install "numpy<2"
```

### 11. Download SpaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 12. Verify GPU Access
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

You should see `CUDA available: True` and your GPU info.

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

**Conda/Environment Issues:**
- If prompt doesn't update after `conda activate`, run `conda init powershell` and restart
- For "not recognized" errors, check your PATH and restart
- If NumPy version errors, downgrade with `pip install "numpy<2"`

**CUDA Out of Memory:**
- Reduce `BATCH_SIZE` to 16 or 8
- Enable gradient checkpointing
- Use `torch.cuda.empty_cache()` between stages

**Slow Data Loading:**
- Use `num_workers=4` (based on your 14 cores)
- Enable `pin_memory=True`

**Installation Issues:**
- If PyTorch CUDA fails: Try `pip install torch --force-reinstall`
- If CUDA is `False`, ensure you installed the **correct PyTorch CUDA wheel** as above

## Summary of Commands

```sh
# One-time setup
conda init powershell

# For each project session
conda activate tiny-diffusion
python --version    # Should be 3.10.x
pip install -r requirements.txt
```

## Updated Requirements.txt

For CUDA users, comment out or remove torch lines and install manually first:

```txt
# Core ML Framework (comment out these lines if you installed manually above)
# torch>=2.1.0
# torchvision>=0.16.0
# torchaudio>=2.1.0
# --extra-index-url https://download.pytorch.org/whl/cu121

# Transformers & NLP
transformers>=4.36.0
tokenizers>=0.15.0
datasets>=2.15.0
sentence-transformers>=2.2.2

# Training & Optimization
accelerate>=0.25.0
wandb>=0.16.0

# Scientific Computing
numpy<2
scipy>=1.11.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Text Processing & Analysis
nltk>=3.8.1
spacy>=3.7.0
textstat>=0.7.3
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm>=4.66.0
pyyaml>=6.0
jupyter>=1.0.0
ipykernel>=6.25.0
einops>=0.7.0

# Evaluation & Metrics
evaluate>=0.4.0
rouge-score>=0.1.2
bert-score>=0.3.13

# Development & Debugging
pytest>=7.4.0
black>=23.9.0
flake8>=6.1.0

# Optional: Advanced features (install if needed)
# faiss-cpu>=1.7.4
# umap-learn>=0.5.4
# deepspeed>=0.12.0
# bitsandbytes>=0.41.0
```