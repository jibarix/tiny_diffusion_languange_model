# 🧹 Project Cleanup Guide

The `cleanup.py` script helps you clean up project files for fresh training runs.

## Quick Usage

```bash
# Quick clean (most common) - removes outputs and cache
python cleanup.py --quick

# Deep clean - removes everything except source code  
python cleanup.py --deep

# Dry run - see what would be deleted without deleting
python cleanup.py --dry-run

# Auto-confirm without prompts
python cleanup.py --quick --yes
```

## Custom Cleanup

```bash
# Clean specific targets
python cleanup.py --targets outputs data
python cleanup.py --targets cache tensorboard
python cleanup.py --targets all

# Available targets:
#   outputs     - Model checkpoints, results, logs
#   data        - Processed datasets, tokenizers, embeddings  
#   cache       - Python cache, temporary files
#   tensorboard - Tensorboard logs
#   logs        - All log files
#   all         - Everything except source code
```

## What Gets Cleaned

### 📁 **Outputs** (Training Results)
- `outputs/` - Model checkpoints, results
- `runs/` - Training runs
- `logs/` - Training logs  
- `checkpoints/` - Model checkpoints
- `models/` - Saved models
- `results/` - Evaluation results

### 💾 **Processed Data**
- `data/processed/` - Processed datasets
- `data/cache/` - Cached data
- `data/embeddings/` - Sentence embeddings
- `data/tokenizers/` - Custom tokenizers
- `data/*.pkl` - Pickle files (segments, splits)
- `data/*.npy` - NumPy arrays
- `data/*.pt/.pth` - PyTorch tensors

### 🗂️ **Cache Files**
- `cache/` - General cache
- `__pycache__/` - Python bytecode
- `.pytest_cache/` - Test cache
- `*.pyc, *.pyo` - Compiled Python
- Temporary files (`*~`, `.DS_Store`, etc.)

### 📊 **Tensorboard Logs**
- `tensorboard/` - Tensorboard logs
- `tb_logs/` - Tensorboard logs
- `runs/*/tensorboard` - Nested TB logs
- `outputs/*/tensorboard` - Output TB logs

## Protected Files

The script **NEVER** deletes:
- Source code (`src/`, `config/`, `scripts/`)
- Raw data (`data/raw/`)
- Documentation (`README.md`, etc.)
- Git repository (`.git/`, `.gitignore`)
- Configuration files (`config.yaml`, etc.)

## Common Workflows

### 🚀 **Starting New Experiment**
```bash
# Quick clean before new training
python cleanup.py --quick --yes
python scripts/prepare_data.py --book data/raw/frankenstein.txt
python scripts/train.py
```

### 🔬 **Fresh Data Processing**
```bash
# Clean processed data to re-run pipeline
python cleanup.py --targets data --yes
python scripts/prepare_data.py --book data/raw/frankenstein.txt
```

### 🧪 **Clean Everything for Publishing**
```bash
# Deep clean before committing to repo
python cleanup.py --deep
```

### 🔍 **Check What Would Be Deleted**
```bash
# See cleanup plan without deleting
python cleanup.py --targets all --dry-run
```

## Safety Features

- **Preview before deletion** - Shows exactly what will be deleted
- **Confirmation prompts** - Asks before deleting (unless `--yes`)
- **Protected files** - Never deletes source code or raw data
- **Size reporting** - Shows how much space will be freed
- **Error handling** - Reports files that couldn't be deleted
- **Dry run mode** - Preview without changes

## Example Output

```
🧹 CLEANUP PLAN
==================================================

📁 OUTPUTS:
   📂 outputs/
   📂 runs/experiment_1/
   📄 model_checkpoint.pt (245.3 MB)

📁 CACHE:
   📂 __pycache__/
   📄 data/segments.pkl (12.4 MB)

📊 SUMMARY:
   Items to delete: 4
   Total size: 257.7 MB

⚠️  DELETE 4 items? [y/N]: y

🗑️  Deleting files...
   🗂️  Deleted directory: outputs
   🗂️  Deleted directory: runs/experiment_1
   📄 Deleted file: model_checkpoint.pt
   🗂️  Deleted directory: __pycache__

✅ Cleanup complete!
   Deleted: 4 items
   Freed: 257.7 MB

🎉 Cleanup successful! Ready for fresh training.
```

## Tips

1. **Always use `--dry-run` first** to see what will be deleted
2. **Use `--quick` for routine cleanup** between experiments  
3. **Use `--deep` only when starting completely fresh**
4. **Keep raw data in `data/raw/`** - it's protected from deletion
5. **The script is safe** - it won't delete source code or git history
