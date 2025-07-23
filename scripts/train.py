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
from config import ProjectConfig, ModelConfig, TrainingConfig, CurriculumConfig


sys.path.append(str(Path(__file__).parent.parent))

from config import ProjectConfig
from src.model.diffusion import MaskedDiffusionLM
from src.training.trainer import CurriculumTrainer
from src.training.scheduler import CurriculumScheduler


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_data(data_dir: str, val_split: float = 0.1):
    """Load processed data and create train/val splits"""
    data_path = Path(data_dir)
    
    # Load curriculum splits
    with open(data_path / "curriculum_splits.pkl", "rb") as f:
        curriculum_splits = pickle.load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_path / "tokenizer")
    
    # Create validation split from all data
    all_segments = curriculum_splits['all']
    random.shuffle(all_segments)
    
    val_size = int(len(all_segments) * val_split)
    val_data = all_segments[:val_size]
    
    # Remove validation data from training splits
    train_indices = set(range(val_size, len(all_segments)))
    
    train_splits = {}
    for split_name, segments in curriculum_splits.items():
        # Filter out validation segments
        if split_name == 'all':
            train_splits[split_name] = all_segments[val_size:]
        else:
            # For other splits, filter based on indices in 'all' split
            segment_indices = {id(seg): i for i, seg in enumerate(all_segments)}
            filtered_segments = []
            for seg in segments:
                if id(seg) in segment_indices:
                    idx = segment_indices[id(seg)]
                    if idx in train_indices:
                        filtered_segments.append(seg)
            train_splits[split_name] = filtered_segments
    
    return train_splits, val_data, tokenizer


def create_model(config: ProjectConfig, tokenizer) -> MaskedDiffusionLM:
    """Create and initialize the diffusion model"""
    model = MaskedDiffusionLM(
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
    
    print(f"Model created: {model.get_num_params():,} parameters")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train masked diffusion language model")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--data-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
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
    output_dir = Path(args.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting training: {config.experiment_name}")
    print(f"📁 Output directory: {output_dir}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️  No GPU available, using CPU")
    
    # Load data
    print("📚 Loading training data...")
    train_data, val_data, tokenizer = load_data(args.data_dir, config.training.val_split)
    
    print(f"Training splits:")
    for split_name, segments in train_data.items():
        print(f"  {split_name}: {len(segments)} examples")
    print(f"Validation: {len(val_data)} examples")
    
    # Create model
    print("🏗️  Creating model...")
    model = create_model(config, tokenizer)
    
    # Create scheduler
    curriculum_scheduler = CurriculumScheduler(config.curriculum)
    
    # Print curriculum summary
    summary = curriculum_scheduler.get_stage_summary()
    print(f"\n📋 Curriculum Summary:")
    print(f"Stages: {summary['total_stages']}, Total epochs: {summary['total_epochs']}")
    for stage_info in summary['stages']:
        print(f"  Stage {stage_info['index'] + 1}: {stage_info['name']}")
        print(f"    Epochs: {stage_info['epochs']}, Masking: {stage_info['masking_range']}")
        print(f"    Data: {stage_info['data_selection']}, Format: {stage_info['format_type']}")
    
    # Create trainer
    trainer = CurriculumTrainer(
        model=model,
        curriculum_scheduler=curriculum_scheduler,
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer,
        config=config,
        output_dir=str(output_dir)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Save configuration
    config.save_yaml(output_dir / "config.yaml")
    print(f"💾 Configuration saved to: {output_dir / 'config.yaml'}")
    
    # Start training
    try:
        trainer.train()
        print("✅ Training completed successfully!")
        
        # Print final summary
        summary = trainer.metrics.get_progress_summary()
        print(f"\n📊 Final Results:")
        print(f"Best validation loss: {summary['best_val_loss']:.4f}")
        print(f"Best validation perplexity: {summary['best_val_perplexity']:.2f}")
        print(f"Total training time: {summary['total_time'] / 3600:.1f} hours")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        trainer.save_checkpoint()
        print("💾 Current state saved")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        trainer.save_checkpoint()
        print("💾 Emergency checkpoint saved")
        raise


if __name__ == "__main__":
    main()