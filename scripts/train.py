#!/usr/bin/env python3
"""
Main Training Script - Tiny Diffusion Text Generation Project

This script provides a unified entry point for the complete training workflow:
- Integrated data preparation with curriculum learning
- 3-stage training progression (foundational → structural → refinement)
- Built-in evaluation and checkpoint management
- Debug and testing modes for development

Usage:
    python scripts/train.py --book data/raw/frankenstein.txt          # Full training
    python scripts/train.py --book data/raw/frankenstein.txt --debug # Debug mode
    python scripts/train.py --test                                    # Quick test
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.backends.cudnn as cudnn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import ProjectConfig
from src.data import DataPipeline, create_debug_data_pipeline
from src.trainer import create_trainer_from_config, test_trainer, quick_training_test, estimate_training_time
from src.evaluation import EvaluationSuite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', encoding='utf-8')  # Add UTF-8 encoding
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


def setup_environment():
    """Setup training environment and hardware optimizations."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        cudnn.benchmark = True  # Optimize for consistent input sizes
        cudnn.deterministic = False  # Allow some non-determinism for speed
    
    # Memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")


def test_installation():
    """Ultra-fast test to verify installation and basic functionality."""
    logger.info("Running installation test...")
    
    try:
        # Test config system
        config = ProjectConfig.debug()
        logger.info("[OK] Configuration system working")
        
        # Test data pipeline creation
        pipeline = create_debug_data_pipeline(config.to_dict())
        logger.info("[OK] Data pipeline creation working")
        
        # Test trainer creation
        trainer = create_trainer_from_config(config.to_dict(), pipeline, device='cpu', debug_mode=True)
        logger.info("[OK] Trainer creation working")
        
        # Setup stage to initialize model
        trainer._setup_stage(0)
        logger.info("[OK] Stage setup working")
        
        # Test model forward pass with valid token IDs
        batch_size, seq_len = 2, 32
        vocab_size = len(pipeline.tokenizer.compressed_vocab)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))  # Use actual vocab size
        attention_mask = torch.ones_like(input_ids)

        # Add debugging here:
        print(f"Model config: {trainer.config['model']}")
        print(f"Input shape: {input_ids.shape}")
        print(f"Vocab size: {vocab_size}")
        print(f"d_model: {trainer.config['model'].get('d_model')}")
        print(f"n_heads: {trainer.config['model'].get('n_heads')}")
        print(f"head_dim: {trainer.config['model'].get('head_dim')}")
        
        trainer.model.eval()
        with torch.no_grad():
            outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask)
            logger.info("[OK] Model forward pass working")
        
        logger.info("Installation test PASSED! All components working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"[FAILED] Installation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """
    Integration test that runs the full pipeline with minimal data.
    
    This takes ~30 seconds and verifies end-to-end functionality.
    """
    logger.info("Running integration test...")
    
    try:
        # Create test configuration
        config = ProjectConfig.debug()
        config_dict = config.to_dict()
        
        # Further reduce for testing
        config_dict['curriculum']['stages'][0]['epochs'] = 1
        config_dict['curriculum']['stages'][1]['epochs'] = 1
        config_dict['curriculum']['stages'][2]['epochs'] = 1
        config_dict['training']['batch_size'] = 2
        
        # Create test data pipeline
        pipeline = create_debug_data_pipeline(config_dict)
        pipeline.print_curriculum_summary()
        
        # Test trainer
        test_trainer(config_dict, pipeline)
        
        # Quick training test
        quick_training_test(config_dict, pipeline, max_steps=3)
        
        logger.info("[OK] Integration test PASSED! Full pipeline working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"[FAILED] Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def prepare_data(book_path: str, config: Dict[str, Any]) -> DataPipeline:
    """
    Prepare training data with curriculum learning.
    
    Args:
        book_path: Path to the source text file
        config: Training configuration
        
    Returns:
        Configured DataPipeline ready for training
    """
    logger.info(f"Preparing data from: {book_path}")
    
    # Validate input file
    if not Path(book_path).exists():
        raise FileNotFoundError(f"Book file not found: {book_path}")
    
    # Create data pipeline
    pipeline = DataPipeline(config)
    
    # Process the book (this does loading, segmentation, scoring, curriculum construction)
    logger.info("Processing book with curriculum construction...")
    pipeline.process_book(book_path, save_dir="data/processed")
    
    # Print summary
    pipeline.print_curriculum_summary()
    
    logger.info("Data preparation complete!")
    return pipeline


def run_training(config: Dict[str, Any], data_pipeline: DataPipeline, resume_from: Optional[str] = None, debug_mode: bool = False):
    """
    Execute the complete 3-stage training curriculum.
    
    Args:
        config: Training configuration
        data_pipeline: Prepared data pipeline
        resume_from: Optional checkpoint path to resume from
        debug_mode: Flag to isolate debug outputs
    """
    logger.info("Starting training...")
    
    # Print training estimates
    estimate_training_time(config, data_pipeline)
    
    # Create trainer, passing the debug_mode flag
    trainer = create_trainer_from_config(config, data_pipeline, device='auto', debug_mode=debug_mode)
    
    # Resume from checkpoint if specified
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.resume_from_checkpoint(resume_from)
    
    # Execute curriculum training
    logger.info("Beginning 3-stage curriculum training...")
    stage_results = trainer.train_full_curriculum()
    
    # Return best model path (constructed from trainer's checkpoint directory)
    best_model_path = trainer.checkpoint_dir / "best_stage3.pt"
    logger.info(f"Training complete! Best model saved to: {best_model_path}")
    return str(best_model_path)


def run_evaluation(model_path: str, config: Dict[str, Any], data_pipeline: DataPipeline):
    """
    Run comprehensive evaluation on the trained model.
    
    Args:
        model_path: Path to the trained model checkpoint
        config: Training configuration  
        data_pipeline: Data pipeline for evaluation
    """
    logger.info("Running evaluation...")
    
    try:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        from src.model import load_model_checkpoint
        model, checkpoint = load_model_checkpoint(model_path, device=device_str)
        
        tokenizer = data_pipeline.tokenizer
        if tokenizer is None:
            logger.warning("No tokenizer available, skipping evaluation")
            return

        # Use the full evaluation suite
        suite = EvaluationSuite(model, tokenizer, data_pipeline, device=device_str)
        
        # Run the full evaluation
        results = suite.full_evaluation(reference_text=data_pipeline.raw_text)
        
        # --- MODIFICATION: Log generated text samples to console ---
        logger.info("Generated sample texts:")
        if 'generation_test' in results and 'results' in results['generation_test']:
            for gen_result in results['generation_test']['results']:
                logger.info("-" * 50)
                logger.info(f"Prompt: '{gen_result.prompt}'")
                logger.info(f"Generated: {gen_result.generated_text}")
        logger.info("-" * 50)
        # --- END MODIFICATION ---

        # Print the summary table
        suite.print_evaluation_summary(results)
        
        logger.info("Evaluation complete!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        logger.info("Skipping evaluation - training was successful though!")


def main():
    """Main training script entry point."""
    import signal
    def signal_handler(sig, frame):
        print('\n Training interrupted by user')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(
        description='Tiny Diffusion Text Generation Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training with debug output
  python scripts/train.py --book data/raw/frankenstein.txt --debug
  
  # Memory-efficient training
  python scripts/train.py --book data/raw/frankenstein.txt --memory-efficient
  
  # Quick installation test
  python scripts/train.py --test
  
  # Integration test
  python scripts/train.py --test-integration
  
  # Resume training from checkpoint
  python scripts/train.py --book data/raw/frankenstein.txt --resume outputs/checkpoint_epoch_50.pt
        """
    )
    
    # Input options
    parser.add_argument('--book', type=str, help='Path to the book text file')
    parser.add_argument('--resume', type=str, help='Resume training from checkpoint')
    
    # Training modes
    parser.add_argument('--debug', action='store_true', help='Debug mode (1 epoch per stage)')
    parser.add_argument('--memory-efficient', action='store_true', help='Use memory-efficient settings')
    
    # Testing options
    parser.add_argument('--test', action='store_true', help='Run quick installation test (~10s)')
    parser.add_argument('--test-integration', action='store_true', help='Run integration test (~30s)')
    
    # Configuration overrides
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    parser.add_argument('--epochs', type=str, help='Override epochs per stage (e.g., "50,75,100")')
    parser.add_argument('--label-smoothing', type=float, help='Label smoothing factor')
    parser.add_argument('--override', action='append', help='Override config value: key=value')

    # Output options
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--no-eval', action='store_true', help='Skip evaluation after training')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Handle test modes
    if args.test:
        success = test_installation()
        sys.exit(0 if success else 1)
    
    if args.test_integration:
        success = test_integration()
        sys.exit(0 if success else 1)
    
    # Validate arguments for training
    if not args.book and not args.resume:
        parser.error("Must specify --book for training or --test for testing")
    
    try:
        # Create configuration
        if args.debug:
            config = ProjectConfig.debug()
            logger.info("Using debug configuration (fast training)")
        elif args.memory_efficient:
            config = ProjectConfig.memory_efficient()
            logger.info("Using memory-efficient configuration")
        else:
            config = ProjectConfig.default()
            logger.info("Using default configuration")
        
        # Apply command-line overrides
        overrides = {}
        if args.batch_size:
            overrides['training.batch_size'] = args.batch_size
        if args.learning_rate:
            overrides['training.learning_rate'] = args.learning_rate
        if args.epochs:
            epoch_list = [int(x.strip()) for x in args.epochs.split(',')]
            if len(epoch_list) == 3:
                for i, epochs in enumerate(epoch_list):
                    overrides[f'curriculum.stages[{i}].epochs'] = epochs
        if args.label_smoothing:
            overrides['training.label_smoothing'] = args.label_smoothing
        if args.output_dir:
            overrides['training.output_dir'] = args.output_dir

        # Handle generic overrides
        if hasattr(args, 'override') and args.override:
            for override_str in args.override:
                key, value = override_str.split('=', 1)
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass  # Keep as string
                overrides[key] = value

        if overrides:
            config = config.override(**overrides)
            logger.info(f"Applied overrides: {overrides}")
        
        config_dict = config.to_dict()
        
        # Prepare data (unless resuming)
        if args.book:
            data_pipeline = prepare_data(args.book, config_dict)
        else:
            # For resume-only, create minimal pipeline
            # In practice, you'd want to save/load the pipeline state
            logger.warning("Resuming without data preparation - using debug pipeline")
            data_pipeline = create_debug_data_pipeline(config_dict)
        
        # Run training
        best_model_path = run_training(config_dict, data_pipeline, resume_from=args.resume, debug_mode=args.debug)
        
        # Run evaluation (unless disabled)
        if not args.no_eval:
            run_evaluation(best_model_path, config_dict, data_pipeline)
        
        logger.info("[OK] Training script completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
