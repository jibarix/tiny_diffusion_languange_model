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

from config import ProjectConfig, ModelConfig, TrainingConfig, CurriculumConfig, GenerationConfig, load_config_with_overrides
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
    """Create model from config with proper vocab size handling"""
    
    # Check vocab size mismatch
    tokenizer_vocab_size = len(tokenizer)
    config_vocab_size = config.model.vocab_size
    
    if tokenizer_vocab_size != config_vocab_size:
        print(f"⚠️  Vocab size mismatch:")
        print(f"   Tokenizer size: {tokenizer_vocab_size:,}")
        print(f"   Config size: {config_vocab_size:,}")
        
        # Use the tokenizer size (what actually exists)
        actual_vocab_size = tokenizer_vocab_size
        print(f"   Using tokenizer size: {actual_vocab_size:,}")
        print(f"   💡 To fix: Update model_config.py vocab_size to {actual_vocab_size}")
    else:
        actual_vocab_size = config_vocab_size
        print(f"✅ Vocab size matches: {actual_vocab_size:,}")
    
    # Validate tokenizer has required special tokens
    required_tokens = ['pad_token_id', 'mask_token_id']
    missing_tokens = []
    for token_attr in required_tokens:
        if getattr(tokenizer, token_attr) is None:
            missing_tokens.append(token_attr)
    
    if missing_tokens:
        raise ValueError(f"Tokenizer missing required tokens: {missing_tokens}. "
                        f"Please ensure tokenizer has special tokens.")
    
    print(f"🏗️  Creating model with {actual_vocab_size:,} vocab size...")
    
    return MaskedDiffusionLM(
        vocab_size=actual_vocab_size,          # Use actual tokenizer size
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
    """Load processed data and create train/val splits with compressed tokenizer support"""
    import json
    data_path = Path(data_dir)
    
    # Load curriculum splits
    with open(data_path / "curriculum_splits.pkl", "rb") as f:
        curriculum_splits = pickle.load(f)
    
    # Load tokenizer with compression support
    tokenizer = None
    tokenizer_path = data_path / "tokenizer"
    
    if tokenizer_path.exists():
        try:
            # Check if this is a compressed tokenizer
            compression_info_path = tokenizer_path / "compression_info.json"
            
            if compression_info_path.exists():
                print("🔧 Loading compressed tokenizer...")
                
                # Load base tokenizer
                base_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                
                # Load compression info
                with open(compression_info_path, 'r') as f:
                    compression_info = json.load(f)
                
                # Recreate CompressedTokenizer class (same as in pipeline)
                class CompressedTokenizer:
                    def __init__(self, base_tokenizer, compression_info):
                        self.base_tokenizer = base_tokenizer
                        
                        # Convert string keys back to integers
                        self.old_to_new = {int(k): v for k, v in compression_info['old_to_new_mapping'].items()}
                        self.new_to_old = {int(k): v for k, v in compression_info['new_to_old_mapping'].items()}
                        self.new_vocab = compression_info['new_vocab']
                        self.vocab_size = compression_info['vocab_size']
                        
                        # Set special tokens
                        special = compression_info['special_tokens']
                        self.pad_token_id = special['pad_token_id']
                        self.mask_token_id = special['mask_token_id']
                        self.bos_token_id = special.get('bos_token_id')
                        self.eos_token_id = special.get('eos_token_id')
                        self.unk_token_id = special.get('unk_token_id', 0)
                        
                        # Set special token strings
                        self.pad_token = base_tokenizer.pad_token
                        self.mask_token = base_tokenizer.mask_token
                        self.bos_token = base_tokenizer.bos_token
                        self.eos_token = base_tokenizer.eos_token
                        self.unk_token = base_tokenizer.unk_token
                    
                    def encode(self, text, add_special_tokens=True, **kwargs):
                        base_ids = self.base_tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)
                        compressed_ids = []
                        
                        for token_id in base_ids:
                            if token_id in self.old_to_new:
                                compressed_ids.append(self.old_to_new[token_id])
                            else:
                                compressed_ids.append(self.unk_token_id)
                        
                        return compressed_ids
                    
                    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
                        base_ids = []
                        for token_id in token_ids:
                            if token_id in self.new_to_old:
                                base_ids.append(self.new_to_old[token_id])
                            else:
                                base_ids.append(self.base_tokenizer.unk_token_id or 0)
                        
                        return self.base_tokenizer.decode(base_ids, skip_special_tokens=skip_special_tokens, **kwargs)
                    
                    def __len__(self):
                        return self.vocab_size
                
                # Create compressed tokenizer
                tokenizer = CompressedTokenizer(base_tokenizer, compression_info)
                print(f"✅ Loaded compressed tokenizer: {len(tokenizer):,} tokens")
                
            else:
                # Standard tokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                print(f"✅ Loaded standard tokenizer: {len(tokenizer):,} tokens")
                
        except Exception as e:
            print(f"❌ Failed to load tokenizer: {e}")
            tokenizer = None
    
    # Fallback to GPT-2 if loading failed
    if tokenizer is None:
        print("⚠️  Using GPT-2 fallback tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        special_tokens = {"pad_token": "<pad>", "mask_token": "<mask>", 
                         "bos_token": "<bos>", "eos_token": "<eos>"}
        tokens_to_add = {k: v for k, v in special_tokens.items() 
                        if getattr(tokenizer, k) is None}
        if tokens_to_add:
            tokenizer.add_special_tokens(tokens_to_add)
        print(f"✅ Using GPT-2 fallback: {len(tokenizer):,} tokens")
    
    # Create validation split
    all_segments = curriculum_splits.get('stage_1', []) + curriculum_splits.get('stage_2', []) + curriculum_splits.get('stage_3', [])
    if not all_segments:
        all_segments = curriculum_splits.get('all', [])
    
    random.shuffle(all_segments)
    val_size = int(len(all_segments) * val_split)
    val_data = all_segments[:val_size]
    
    return curriculum_splits, val_data, tokenizer


def create_test_data():
    """Create minimal test data for ultra-fast testing - REVISED VERSION"""
    from collections import Counter
    test_text = """
Natural selection acts by preservation of beneficial variations. Each modification profitable to the organism.
The theory of evolution explains the origin of species through gradual change. Evidence supports this view.
However critics argue against this theory. I contend the evidence is overwhelming.
Frankenstein was written by Mary Shelley. The creature demands a companion from his creator.
The novel explores themes of creation and responsibility. Science fiction emerged from this work.
    """.strip()
    
    # Create minimal pipeline with vocab curriculum DISABLED
    pipeline = TextDataPipeline(
        n_clusters=2, 
        target_vocab_size=1000,  # Small for testing
        enable_argument_mining=True, 
        enable_vocab_curriculum=False  # DISABLED - this was causing issues
    )
    
    # Process into segments manually for speed
    sentences = [s.strip() for s in test_text.split('.') if s.strip()]
    segments = []
    
    for i, sentence in enumerate(sentences):
        # Create realistic difficulty scores
        lexical_score = 0.3 + (i * 0.1) % 0.7  # Varies between 0.3-0.7
        syntactic_score = 0.2 + (i * 0.15) % 0.6  # Varies between 0.2-0.6
        centrality_score = 0.4 + (i * 0.1) % 0.5  # Varies between 0.4-0.7
        
        combined_difficulty = (lexical_score + syntactic_score + centrality_score) / 3.0
        
        # Assign stages based on difficulty (like real curriculum)
        if combined_difficulty < 0.4:
            stage = 1  # Foundation (easy)
        elif combined_difficulty < 0.6:
            stage = 2  # Structural (medium)
        else:
            stage = 3  # Refinement (hard)
        
        segment = TextSegment(
            text=sentence,
            index=i,
            lexical_rarity=lexical_score,
            syntactic_complexity=syntactic_score,
            thematic_centrality=centrality_score,
            combined_difficulty=combined_difficulty,
            stage_assignment=stage,
            vocabulary_level=1,  # All use vocab level 1 (simple)
            length=len(sentence.split()),
            cluster_id=i % 2,  # Alternate between 2 clusters
            argumentative_role="claim" if i % 2 == 0 else "evidence",
            argumentative_confidence=0.7
        )
        segments.append(segment)
    
    # Set segments in pipeline
    pipeline.segments = segments
    
    # Create a simple tokenizer to avoid 50,259 token issues
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Add required special tokens
    special_tokens = {
        "pad_token": "<pad>", 
        "mask_token": "<mask>", 
        "bos_token": "<bos>", 
        "eos_token": "<eos>"
    }
    
    tokens_to_add = {k: v for k, v in special_tokens.items() 
                    if getattr(tokenizer, k) is None}
    
    if tokens_to_add:
        tokenizer.add_special_tokens(tokens_to_add)
    
    # Set tokenizer in pipeline
    pipeline.tokenizer = tokenizer
    
    print(f"✅ Test data created:")
    print(f"   Segments: {len(segments)}")
    print(f"   Stage distribution: {Counter(s.stage_assignment for s in segments)}")
    print(f"   Tokenizer size: {len(tokenizer):,}")
    
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
    parser.add_argument("--config", default=None, help="Config file path")
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
        config = ProjectConfig.debug()
        print("🐛 Fast debug mode: 1 epoch per stage")
    else:
        config = load_config_with_overrides(
            config_file=args.config if args.config else None,
            args=args
        )

    # Validate configuration
    config.validate()

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