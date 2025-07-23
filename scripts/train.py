#!/usr/bin/env python3
"""
Training Script
Entry point for curriculum-based diffusion model training
"""

import argparse
import sys
import os
import random
import pickle
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer

# Add src to path (matches prepare_data.py)
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from config import ProjectConfig, ModelConfig, TrainingConfig, CurriculumConfig
from model.diffusion import MaskedDiffusionLM
from training.trainer import CurriculumTrainer
from training.scheduler import CurriculumScheduler
from data.pipeline import TextSegment


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_model(config: ProjectConfig, tokenizer) -> MaskedDiffusionLM:
    """Create model from config"""
    return MaskedDiffusionLM(
        vocab_size=len(tokenizer),
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        attention_dropout=config.model.attention_dropout,
        use_bias=config.model.use_bias,
        norm_eps=config.model.norm_eps,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id
    )


def load_data(data_dir: str, val_split: float = 0.1):
    """Load processed data and create train/val splits"""
    data_path = Path(data_dir)
    
    # Load curriculum splits
    with open(data_path / "curriculum_splits.pkl", "rb") as f:
        curriculum_splits = pickle.load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_path / "tokenizer")
    
    # Create validation split from all data (take from end to avoid bias)
    all_segments = curriculum_splits['all']
    random.shuffle(all_segments)
    
    val_size = int(len(all_segments) * val_split)
    val_data = all_segments[:val_size]
    train_all = all_segments[val_size:]
    
    # Create training splits
    train_splits = {'all': train_all}
    
    # For other splits, recreate them from the filtered training data
    sorted_train = sorted(train_all, key=lambda x: x.combined_difficulty)
    n_train = len(sorted_train)
    easy_end = n_train // 3
    medium_end = 2 * n_train // 3
    
    train_splits['easy'] = sorted_train[:easy_end]
    train_splits['medium'] = sorted_train[easy_end:medium_end]
    train_splits['hard'] = sorted_train[medium_end:]
    
    return train_splits, val_data, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train curriculum diffusion model")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--data-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Use debug config")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.debug:
        config = ProjectConfig(
            model=ModelConfig.tiny_125m(),
            training=TrainingConfig.fast_debug(), 
            curriculum=CurriculumConfig.fast_debug()
        )
        print("🐛 Debug mode: Using fast debug configuration")
    else:
        config = ProjectConfig.from_yaml(args.config)
    
    # Set seed
    set_seed(config.seed)
    
    # Setup output directory
    output_dir = Path(args.output) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting training: {config.experiment_name}")
    print(f"📁 Output directory: {output_dir}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    
    # Load data
    print("📚 Loading data...")
    train_splits, val_data, tokenizer = load_data(args.data_dir)
    
    print(f"✅ Data loaded:")
    for split_name, segments in train_splits.items():
        print(f"  - {split_name}: {len(segments)} segments")
    print(f"  - validation: {len(val_data)} segments")
    
    # Create model
    print("🏗️ Creating model...")
    model = create_model(config, tokenizer)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create curriculum scheduler
    curriculum_scheduler = CurriculumScheduler(config.curriculum)
    
    # Create trainer
    trainer = CurriculumTrainer(
        model=model,
        curriculum_scheduler=curriculum_scheduler,
        train_data=train_splits,
        val_data=val_data,
        tokenizer=tokenizer,
        config=config,
        output_dir=str(output_dir)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"🔄 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("🎯 Starting curriculum training...")
    trainer.train()
    
    print("✅ Training completed!")


if __name__ == "__main__":
    main()