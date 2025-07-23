#!/usr/bin/env python3
"""
Enhanced Training Script with Fast Testing
Entry point for curriculum-based diffusion model training
"""

import argparse
import sys
import os
import random
import pickle
import tempfile
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from config import ProjectConfig, ModelConfig, TrainingConfig, CurriculumConfig
from model.diffusion import MaskedDiffusionLM
from training.trainer import EnhancedCurriculumTrainer
from data.pipeline import TextDataPipeline, TextSegment


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
    
    # Create validation split from all data
    all_segments = curriculum_splits.get('stage_1', []) + curriculum_splits.get('stage_2', []) + curriculum_splits.get('stage_3', [])
    if not all_segments:
        all_segments = curriculum_splits.get('all', [])
    
    random.shuffle(all_segments)
    
    val_size = int(len(all_segments) * val_split)
    val_data = all_segments[:val_size]
    
    return curriculum_splits, val_data, tokenizer


def create_test_data():
    """Create minimal test data for ultra-fast testing"""
    test_text = """
Natural selection acts by preservation of beneficial variations. Each modification profitable to the organism.
The theory of evolution explains the origin of species through gradual change. Evidence supports this view.
However critics argue against this theory. I contend the evidence is overwhelming.
    """.strip()
    
    # Create minimal pipeline
    pipeline = TextDataPipeline(n_clusters=2, target_vocab_size=1000, 
                               enable_argument_mining=True, enable_vocab_curriculum=False)
    
    # Process into segments manually for speed
    sentences = [s.strip() for s in test_text.split('.') if s.strip()]
    segments = []
    for i, sentence in enumerate(sentences):
        segment = TextSegment(
            text=sentence,
            index=i,
            lexical_rarity=0.5,
            syntactic_complexity=0.5,
            thematic_centrality=0.5,
            combined_difficulty=0.5,
            stage_assignment=1 if i < 2 else 2,
            length=len(sentence.split())
        )
        segments.append(segment)
    
    pipeline.segments = segments
    return pipeline, segments


def ultra_fast_test():
    """Ultra-fast test - just verify model works"""
    print("🚀 Ultra-fast test (10 seconds)...")
    
    # Minimal config
    config = ProjectConfig(
        model=ModelConfig(d_model=64, n_layers=1, n_heads=1, d_ff=128, max_seq_len=32),
        training=TrainingConfig(batch_size=1),
        curriculum=CurriculumConfig.fast_debug()
    )
    
    # Test tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"mask_token": "<mask>", "pad_token": "<pad>"})
    
    # Test model creation
    model = create_model(config, tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Test forward pass
    x = torch.randint(0, min(1000, len(tokenizer)), (1, 16)).to(device)
    mask = torch.ones_like(x).to(device)
    
    outputs = model(x, attention_mask=mask, masking_rate=0.5)
    loss = model.compute_loss(outputs)
    
    # Test backward pass
    loss.backward()
    
    print(f"✅ Model works! Loss: {loss.item():.4f}, Params: {model.get_num_params():,}")
    return True


def fast_integration_test():
    """Fast integration test with curriculum"""
    print("🧪 Fast integration test (30 seconds)...")
    
    pipeline, segments = create_test_data()
    
    # Minimal config
    config = ProjectConfig(
        model=ModelConfig(d_model=64, n_layers=1, n_heads=1, d_ff=128, max_seq_len=32),
        training=TrainingConfig(batch_size=1, log_every=1),
        curriculum=CurriculumConfig(stages=[
            CurriculumConfig.fast_debug().stages[0].__class__(
                name=CurriculumConfig.fast_debug().stages[0].name,
                epochs=1,
                masking_rate_range=CurriculumConfig.fast_debug().stages[0].masking_rate_range,
                data_selection=CurriculumConfig.fast_debug().stages[0].data_selection,
                format_type=CurriculumConfig.fast_debug().stages[0].format_type
            )
        ])
    )
    
    # Test model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"mask_token": "<mask>", "pad_token": "<pad>"})
    model = create_model(config, tokenizer)
    
    # Test trainer setup
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = EnhancedCurriculumTrainer(
            model=model,
            pipeline=pipeline,
            val_data=segments[:2],
            config=config,
            output_dir=temp_dir
        )
        
        # Test dataloader
        dataloader = trainer.create_stage_dataloader(stage=1, vocab_level=1, batch_size=1)
        batch = next(iter(dataloader))
        
        # Test training step
        input_ids = batch['input_ids'].to(trainer.device)
        attention_mask = batch['attention_mask'].to(trainer.device)
        
        trainer.optimizer.zero_grad()
        outputs = trainer.model(input_ids, attention_mask, masking_rate=0.5)
        loss = trainer.model.compute_loss(outputs)
        loss.backward()
        trainer.optimizer.step()
        
        print(f"✅ Integration test passed! Training step loss: {loss.item():.4f}")
        
        # Close logger to release file handles (Windows fix)
        for handler in trainer.logger.handlers[:]:
            handler.close()
            trainer.logger.removeHandler(handler)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Train curriculum diffusion model")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--data-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--resume", help="Resume from checkpoint")
    
    # Testing options
    parser.add_argument("--test", action="store_true", help="Ultra-fast test (10s)")
    parser.add_argument("--test-integration", action="store_true", help="Fast integration test (30s)")
    parser.add_argument("--debug", action="store_true", help="Fast debug mode (1 epoch per stage)")
    
    args = parser.parse_args()
    
    # Ultra-fast test
    if args.test:
        ultra_fast_test()
        return
    
    # Fast integration test
    if args.test_integration:
        fast_integration_test()
        return
    
    # Load configuration
    if args.debug:
        config = ProjectConfig(
            model=ModelConfig.tiny_125m(),
            training=TrainingConfig(
                batch_size=8,  # Small batch
                max_epochs=3,
                eval_every=5,
                save_every=20,
                log_every=1
            ),
            curriculum=CurriculumConfig(stages=[
                CurriculumConfig.fast_debug().stages[0].__class__(
                    name="debug_foundation",
                    epochs=1,
                    masking_rate_range=(0.75, 0.90),
                    data_selection="easy",
                    format_type="sentences"
                ),
                CurriculumConfig.fast_debug().stages[1].__class__(
                    name="debug_structural", 
                    epochs=1,
                    masking_rate_range=(0.40, 0.60),
                    data_selection="medium",
                    format_type="pairs"
                ),
            ])
        )
        print("🐛 Fast debug mode: 1 epoch per stage")
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
    try:
        curriculum_splits, val_data, tokenizer = load_data(args.data_dir)
        print(f"✅ Data loaded: {len(val_data)} validation samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("💡 Try running: python scripts/prepare_data.py --book your_book.txt")
        sys.exit(1)
    
    # Load enhanced pipeline
    print("🔧 Loading enhanced pipeline...")
    try:
        pipeline = TextDataPipeline()
        pipeline.load_data(args.data_dir)
        print(f"✅ Pipeline loaded: {len(pipeline.segments)} segments")
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        sys.exit(1)
    
    # Create model
    print("🏗️ Creating model...")
    model = create_model(config, tokenizer)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create enhanced trainer
    trainer = EnhancedCurriculumTrainer(
        model=model,
        pipeline=pipeline,
        val_data=val_data,
        config=config,
        output_dir=str(output_dir)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"🔄 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("🎯 Starting enhanced curriculum training...")
    try:
        trainer.train()
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
        # Save current state
        trainer.save_checkpoint(
            trainer.curriculum_scheduler.current_stage + 1,
            trainer.curriculum_scheduler.current_vocab_level
        )
        print("💾 Current state saved")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()