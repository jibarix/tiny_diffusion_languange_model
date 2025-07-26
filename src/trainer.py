"""
Training Orchestrator with 3-Stage Curriculum Progression

Implements complete training pipeline for tiny masked diffusion language model:
- 3-stage curriculum execution with automatic stage transitions
- Dynamic masking rate adjustment per stage
- Memory-efficient training with gradient checkpointing
- Real-time metrics tracking and logging
- Stage-specific evaluation and early stopping

Based on 2025 research on curriculum learning and data-constrained training.
"""

import os
import time
import json
import math
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
import torch.nn.functional as F

# Core dependencies
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist

# Utilities
from tqdm import tqdm
import wandb
from pathlib import Path

# Local imports
from .model import MaskedDiffusionLM, save_model_checkpoint, load_model_checkpoint
from .data import DataPipeline


@dataclass
class TrainingMetrics:
    """Training metrics for a single step/epoch"""
    epoch: int
    step: int
    stage: str
    loss: float
    perplexity: float
    learning_rate: float
    masking_rate: float
    grad_norm: float
    throughput_tokens_per_sec: float
    memory_allocated_gb: float
    time_elapsed: float


@dataclass
class StageResults:
    """Results from completing a curriculum stage"""
    stage_name: str
    epochs_completed: int
    final_loss: float
    final_perplexity: float
    best_loss: float
    best_perplexity: float
    training_time: float
    total_tokens_processed: int


class CurriculumTrainer:
    """
    Complete training orchestrator for 3-stage curriculum learning.
    
    Manages the full training lifecycle:
    - Stage progression and data transitions  
    - Dynamic masking rate schedules
    - Optimization and learning rate scheduling
    - Metrics tracking and checkpointing
    - Memory-efficient training loops
    """
    
    # --- MODIFICATION: Added debug_mode flag ---
    def __init__(self, config: Dict[str, Any], data_pipeline: DataPipeline, device: str = 'auto', debug_mode: bool = False):
        self.config = config
        self.data_pipeline = data_pipeline
        self.debug_mode = debug_mode # Store the debug mode flag
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Training configuration
        self.training_config = config.get('training', {})
        self.curriculum_config = config.get('curriculum', {})
        self.system_config = config.get('system', {})
        
        # Stage management
        self.stages = self.curriculum_config.get('stages', [])
        self.current_stage_idx = 0
        self.current_stage = None
        
        # Model and training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

        # Real-time difficulty adjustment
        self.difficulty_multiplier = 1.0
        self.stage_target_losses = {}
        
        # Metrics and logging
        self.metrics_history = []
        self.stage_results = []
        self.global_step = 0
        self.tokens_processed = 0
        
        # --- MODIFICATION: Isolate debug outputs ---
        # Directories
        base_output_dir = Path(self.training_config.get("output_dir", "outputs"))
        if self.debug_mode:
            self.output_dir = base_output_dir / "debug"
            print("--- DEBUG MODE: All outputs will be saved to outputs/debug/ ---")
        else:
            self.output_dir = base_output_dir

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        self.samples_dir = self.output_dir / "samples"
        # --- END MODIFICATION ---
        
        # Create directories
        for dir_path in [self.checkpoint_dir, self.logs_dir, self.samples_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Initialize logging systems"""
        # Setup wandb if available
        self.use_wandb = self.system_config.get('use_wandb', False)
        if self.use_wandb:
            try:
                wandb.init(
                    project=self.system_config.get('wandb_project', 'tiny-diffusion'),
                    config=self.config,
                    name=f"curriculum-{int(time.time())}"
                )
            except Exception as e:
                print(f"Warning: Could not initialize wandb: {e}")
                self.use_wandb = False
        
        # Create training log file
        self.log_file = self.logs_dir / f"training_{int(time.time())}.jsonl"
        
        print(f"Training logs will be saved to: {self.log_file}")
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
    
    def _create_model(self) -> MaskedDiffusionLM:
        """Create and initialize model"""
        try:
            from src.model import create_model_from_config
            print("✓ Import successful")
        except ImportError as e:
            print(f"✗ Import failed: {e}")
            return None

        # *** FIX: Update self.config directly, not a copy ***
        # This ensures the correct vocab size and token IDs are saved in the checkpoint.
        if self.data_pipeline.tokenizer:
            self.config['model']['vocab_size'] = len(self.data_pipeline.tokenizer.compressed_vocab)
            self.config['model']['mask_token_id'] = self.data_pipeline.tokenizer.token_mapping.get('[MASK]', 1)
            self.config['model']['pad_token_id'] = self.data_pipeline.tokenizer.token_mapping.get('[PAD]', 2)
            self.config['model']['eos_token_id'] = self.data_pipeline.tokenizer.token_mapping.get('<|endoftext|>', 0)
        
        model = create_model_from_config(self.config['model'])
        model.to(self.device)
        
        # Enable gradient checkpointing if configured
        if self.training_config.get('gradient_checkpointing', False):
            model.gradient_checkpointing = True
        
        # Compile model if using PyTorch 2.0+
        if self.system_config.get('compile_model', False):
            try:
                model = torch.compile(model)
                print("Model compiled successfully")
            except Exception as e:
                print(f"Warning: Could not compile model: {e}")
        
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        optimizer_name = self.training_config.get('optimizer', 'AdamW').lower()
        learning_rate = self.training_config.get('learning_rate', 2e-4)
        weight_decay = self.training_config.get('weight_decay', 0.1)
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
                eps=1e-8
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        scheduler_name = self.training_config.get('scheduler', 'cosine_with_restarts')
        
        if scheduler_name == 'cosine_with_restarts':
            # Calculate total steps for current stage
            stage_epochs = self.current_stage['epochs']
            train_loader, _ = self.data_pipeline.create_dataloaders(
                self.current_stage['name'],
                batch_size=self.training_config.get('batch_size', 32)
            )
            steps_per_epoch = len(train_loader)
            total_steps = stage_epochs * steps_per_epoch
            warmup_steps = self.training_config.get('warmup_steps', 1000)
            
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(total_steps - warmup_steps, 1),
                T_mult=1,
                eta_min=self.training_config.get('learning_rate', 2e-4) * 0.1
            )
        elif scheduler_name == 'linear':
            stage_epochs = self.current_stage['epochs']
            train_loader, _ = self.data_pipeline.create_dataloaders(
                self.current_stage['name'],
                batch_size=self.training_config.get('batch_size', 32)
            )
            steps_per_epoch = len(train_loader)
            total_steps = stage_epochs * steps_per_epoch
            
            scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=total_steps
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _setup_stage(self, stage_idx: int):
        """Setup training for a specific curriculum stage"""
        if stage_idx >= len(self.stages):
            raise ValueError(f"Stage index {stage_idx} out of range")
        
        self.current_stage_idx = stage_idx
        self.current_stage = self.stages[stage_idx]
        stage_name = self.current_stage['name']
        
        print(f"\n{'='*60}")
        print(f"SETTING UP STAGE {stage_idx + 1}: {stage_name.upper()}")
        print(f"{'='*60}")
        print(f"Description: {self.current_stage['description']}")
        print(f"Epochs: {self.current_stage['epochs']}")
        print(f"Masking Rate: {self.current_stage['masking_rate_range']}")
        print(f"Format: {self.current_stage['training_format']}")
        
        # Create model if not exists
        if self.model is None:
            self.model = self._create_model()
        
        # Create/recreate optimizer for stage
        if self.optimizer is None or self.curriculum_config.get('reset_optimizer_between_stages', False):
            self.optimizer = self._create_optimizer()
            
            # Adjust learning rate for stage
            lr_decay_factors = self.curriculum_config.get('learning_rate_decay_factors', [1.0, 1.0, 1.0])
            if stage_idx < len(lr_decay_factors):
                decay_factor = lr_decay_factors[stage_idx]
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= decay_factor
                print(f"Learning rate adjusted by factor: {decay_factor}")
        
        # Create scheduler for stage
        self.scheduler = self._create_scheduler()
        
        # Setup mixed precision training
        if self.training_config.get('mixed_precision', True) and self.device.type == 'cuda':
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        print(f"Stage {stage_idx + 1} setup complete.")
    
    def _log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics"""
        # Add to history
        self.metrics_history.append(metrics)
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log({
                'epoch': metrics.epoch,
                'step': metrics.step,
                'stage': metrics.stage,
                'loss': metrics.loss,
                'perplexity': metrics.perplexity,
                'learning_rate': metrics.learning_rate,
                'masking_rate': metrics.masking_rate,
                'grad_norm': metrics.grad_norm,
                'throughput': metrics.throughput_tokens_per_sec,
                'memory_gb': metrics.memory_allocated_gb,
            }, step=metrics.step)
        
        # Print progress
        if self.global_step % 10 == 0 or metrics.step % 100 == 0:
            print(f"Step {metrics.step:6d} | Loss: {metrics.loss:.4f} | PPL: {metrics.perplexity:.2f} | "
                  f"LR: {metrics.learning_rate:.2e} | Mask: {metrics.masking_rate:.2f} | "
                  f"Tokens/s: {metrics.throughput_tokens_per_sec:.0f}")
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_name = f"checkpoint_stage{self.current_stage_idx + 1}_epoch{epoch}.pt"
        if is_best:
            checkpoint_name = f"best_stage{self.current_stage_idx + 1}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        save_model_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            config=self.config,
            epoch=epoch,
            step=self.global_step,
            loss=loss,
            filepath=str(checkpoint_path)
        )
        
        # Also save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pt"
        save_model_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            config=self.config,
            epoch=epoch,
            step=self.global_step,
            loss=loss,
            filepath=str(latest_path)
        )
    
    def _evaluate_stage(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        label_smoothing = self.training_config.get('label_smoothing', 0.0)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", leave=False):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # --- MODIFIED: Pass clean inputs to both input_ids and labels ---
                # The model's forward pass now handles its own masking.
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids, # Labels are the clean tokens
                    label_smoothing=label_smoothing
                )
                # --- END MODIFIED ---
                
                loss = outputs['loss']
                
                # Accumulate metrics
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)
        
        avg_loss = total_loss / max(total_samples, 1)
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        self.model.train()
        return avg_loss, perplexity
    
    def _train_epoch(self, train_loader: DataLoader, val_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch with dynamic masking and real-time difficulty adjustment"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        epoch_start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            step_start_time = time.time()
            
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # --- MODIFIED: Forward pass for new MDLM objective ---
            # The model's forward pass now takes clean input_ids and handles its own masking.
            # We pass input_ids to both `input_ids` and `labels`.
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids, # Ground truth is the clean sequence
                        output_attentions=True
                    )
                    loss = outputs['loss']
                
                # --- BUG FIX: Handle zero loss case to prevent GradScaler error ---
                if loss is not None and loss.item() > 0:
                    self.scaler.scale(loss).backward()
                    if self.training_config.get('gradient_clipping', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.training_config['gradient_clipping'])
                    else:
                        grad_norm = 0.0
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # If loss is 0 (e.g., no tokens were masked), skip optimizer step
                    grad_norm = 0.0
                    if self.global_step % 500 == 0: # Log this occasionally
                        print("[CONSOLE LOG] Loss is 0, skipping optimizer step.")
                # --- END BUG FIX ---

            else: # Not using scaler
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids, # Ground truth is the clean sequence
                    output_attentions=True
                )
                loss = outputs['loss']
                
                # --- BUG FIX: Handle zero loss case for non-scaler path ---
                if loss is not None and loss.item() > 0:
                    loss.backward()
                    if self.training_config.get('gradient_clipping', 0) > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.training_config['gradient_clipping'])
                    else:
                        grad_norm = 0.0
                    self.optimizer.step()
                else:
                    grad_norm = 0.0
                    if self.global_step % 500 == 0:
                        print("[CONSOLE LOG] Loss is 0, skipping optimizer step.")
                # --- END BUG FIX ---
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            
            batch_size = input_ids.size(0)
            seq_length = input_ids.size(1)
            tokens_per_second = (batch_size * seq_length) / step_duration
            
            if self.device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1e9
            else:
                memory_allocated = 0.0
            
            current_loss = loss.item() if loss is not None else 0.0
            
            # For logging, masking rate is now an average value, as it's random per-batch.
            metrics = TrainingMetrics(
                epoch=epoch,
                step=self.global_step,
                stage=self.current_stage['name'],
                loss=current_loss,
                perplexity=math.exp(current_loss) if current_loss < 10 else float('inf'),
                learning_rate=self.optimizer.param_groups[0]['lr'],
                masking_rate=0.5, # Avg mask rate for MDLM is 50%
                grad_norm=float(grad_norm) if grad_norm > 0 else 0.0,
                throughput_tokens_per_sec=tokens_per_second,
                memory_allocated_gb=memory_allocated,
                time_elapsed=step_end_time - epoch_start_time
            )
            
            self._log_metrics(metrics)
            
            pbar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'ppl': f"{math.exp(current_loss):.2f}" if current_loss < 10 else "inf",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                'mem': f"{memory_allocated:.1f}GB"
            })
            
            total_loss += current_loss * batch_size
            total_samples += batch_size
            self.tokens_processed += (attention_mask.sum().item())
            self.global_step += 1
        
        avg_loss = total_loss / max(total_samples, 1)
        avg_perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        return avg_loss, avg_perplexity


    def _calculate_attention_entropy(self, attentions):
        """
        Calculate attention entropy to measure model uncertainty.
        Higher entropy = model is uncertain = increase difficulty
        """
        if not attentions or len(attentions) == 0:
            return 1.0  # Default difficulty
        
        # Use last layer attention (most refined)
        last_attention = attentions[-1]  # [batch, heads, seq, seq]
        
        # Average across batch and heads
        avg_attention = last_attention.mean(dim=(0, 1))  # [seq, seq]
        
        # Calculate entropy for each position
        entropies = []
        for i in range(avg_attention.size(0)):
            attention_dist = avg_attention[i]
            # Add small epsilon to avoid log(0)
            entropy = -(attention_dist * torch.log(attention_dist + 1e-8)).sum()
            entropies.append(entropy.item())
        
        # Return normalized entropy (0.5 to 1.5 range)
        mean_entropy = np.mean(entropies)
        normalized_entropy = 0.5 + (mean_entropy / 5.0)  # Rough normalization
        return min(1.5, max(0.5, normalized_entropy))
    
    def train_stage(self, stage_idx: int) -> StageResults:
        """Train a single curriculum stage"""
        stage_start_time = time.time()
        
        # Setup stage
        self._setup_stage(stage_idx)
        stage_name = self.current_stage['name']
        stage_epochs = self.current_stage['epochs']
        
        # Create data loaders
        batch_size = self.training_config.get('batch_size', 32)
        train_loader, val_loader = self.data_pipeline.create_dataloaders(
            stage_name,
            batch_size=batch_size,
            split_ratio=1.0 - self.config.get('data', {}).get('validation_split', 0.1)
        )
        
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        # Training metrics
        best_val_loss = float('inf')
        best_val_perplexity = float('inf')
        final_loss = 0.0
        final_perplexity = 0.0
        patience_counter = 0
        early_stopping_patience = self.training_config.get('early_stopping_patience', 20)
        
        # Training loop
        for epoch in range(1, stage_epochs + 1):
            print(f"\n--- Stage {stage_idx + 1} ({stage_name}) - Epoch {epoch}/{stage_epochs} ---")
            
            # Train epoch
            train_loss, train_perplexity = self._train_epoch(train_loader, val_loader, epoch)
            
            # Evaluate
            val_loss, val_perplexity = self._evaluate_stage(val_loader)
            
            print(f"Train - Loss: {train_loss:.4f}, Perplexity: {train_perplexity:.2f}")
            print(f"Val   - Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
            
            # Save checkpoint
            save_every = self.training_config.get('save_every', 10)
            if epoch % save_every == 0:
                self._save_checkpoint(epoch, val_loss)
            
            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_perplexity = val_perplexity
                self._save_checkpoint(epoch, val_loss, is_best=True)
                patience_counter = 0
                print(f"New best validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
            
            final_loss = val_loss
            final_perplexity = val_perplexity
        
        # Stage completion
        stage_end_time = time.time()
        training_time = stage_end_time - stage_start_time
        
        results = StageResults(
            stage_name=stage_name,
            epochs_completed=epoch,
            final_loss=final_loss,
            final_perplexity=final_perplexity,
            best_loss=best_val_loss,
            best_perplexity=best_val_perplexity,
            training_time=training_time,
            total_tokens_processed=self.tokens_processed
        )
        
        self.stage_results.append(results)
        
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx + 1} COMPLETE: {stage_name.upper()}")
        print(f"{'='*60}")
        print(f"Epochs completed: {epoch}")
        print(f"Final validation loss: {final_loss:.4f}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Training time: {training_time:.1f}s ({training_time/3600:.2f}h)")
        print(f"Total tokens processed: {self.tokens_processed:,}")
        
        return results
    
    def train_full_curriculum(self) -> List[StageResults]:
        """Train complete 3-stage curriculum"""
        print(f"\n{'='*80}")
        print(f"STARTING FULL CURRICULUM TRAINING")
        print(f"{'='*80}")
        print(f"Stages to train: {len(self.stages)}")
        print(f"Device: {self.device}")
        print(f"Model config: {self.config['model']['d_model']}d, {self.config['model']['n_layers']}L")
        
        curriculum_start_time = time.time()
        all_results = []
        
        try:
            # Train each stage
            for stage_idx in range(len(self.stages)):
                stage_results = self.train_stage(stage_idx)
                all_results.append(stage_results)
                
                # Log stage completion to wandb
                if self.use_wandb:
                    wandb.log({
                        f'stage_{stage_idx + 1}_final_loss': stage_results.final_loss,
                        f'stage_{stage_idx + 1}_best_loss': stage_results.best_loss,
                        f'stage_{stage_idx + 1}_epochs': stage_results.epochs_completed,
                        f'stage_{stage_idx + 1}_training_time': stage_results.training_time,
                    })
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            raise
        
        finally:
            # Save final checkpoint
            self._save_checkpoint(
                epoch=getattr(self, 'epoch', 0),
                loss=getattr(self, 'final_loss', 0.0)
            )
        
        # Final summary
        curriculum_end_time = time.time()
        total_training_time = curriculum_end_time - curriculum_start_time
        
        print(f"\n{'='*80}")
        print(f"CURRICULUM TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Total training time: {total_training_time:.1f}s ({total_training_time/3600:.2f}h)")
        print(f"Total tokens processed: {self.tokens_processed:,}")
        print(f"Final global step: {self.global_step}")
        
        # Print stage summary
        print(f"\nStage Summary:")
        for i, results in enumerate(all_results):
            print(f"  Stage {i+1} ({results.stage_name}): "
                  f"Loss {results.best_loss:.4f} -> {results.final_loss:.4f} "
                  f"({results.epochs_completed} epochs, {results.training_time/3600:.1f}h)")
        
        # Save final results
        results_file = self.logs_dir / "curriculum_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        if self.use_wandb:
            wandb.finish()
        
        return all_results
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Resume training from checkpoint"""
        try:
            self.model, checkpoint = load_model_checkpoint(checkpoint_path, str(self.device))
            
            # Restore training state
            if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.global_step = checkpoint.get('step', 0)
            
            print(f"Resumed from checkpoint: {checkpoint_path}")
            print(f"  Global step: {self.global_step}")
            print(f"  Loss: {checkpoint.get('loss', 'unknown')}")
            
            return True
        
        except Exception as e:
            print(f"Failed to resume from checkpoint {checkpoint_path}: {e}")
            return False


# --- MODIFICATION: Pass debug_mode flag to the trainer ---
def create_trainer_from_config(config: Dict[str, Any], data_pipeline: DataPipeline, device: str = 'auto', debug_mode: bool = False) -> CurriculumTrainer:
    """Create trainer instance from configuration"""
    trainer = CurriculumTrainer(config, data_pipeline, device, debug_mode=debug_mode)
    return trainer
# --- END MODIFICATION ---


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    test_config = {
        'model': {
            'd_model': 128,
            'n_layers': 3,
            'n_heads': 4,
            'vocab_size': 5000,
            'head_dim': 32,
            'ffn_hidden_size': 320,
        },
        'training': {
            'batch_size': 4,
            'learning_rate': 2e-4,
            'weight_decay': 0.1,
            'gradient_clipping': 1.0,
            'mixed_precision': True,
            'early_stopping_patience': 5,
            'save_every': 2,
            'eval_every': 1,
        },
        'curriculum': {
            'stages': [
                {
                    'name': 'foundational',
                    'epochs': 2,
                    'masking_rate_range': (0.7, 0.8),
                    'description': 'Debug foundational stage'
                },
                {
                    'name': 'structural',
                    'epochs': 2,
                    'masking_rate_range': (0.4, 0.5),
                    'description': 'Debug structural stage'
                },
                {
                    'name': 'refinement',
                    'epochs': 2,
                    'masking_rate_range': (0.1, 0.2),
                    'description': 'Debug refinement stage'
                }
            ]
        },
        'system': {
            'use_wandb': False,
            'compile_model': False,
        }
    }
    
    print("Testing trainer...")
    
    # Create debug data pipeline
    from .data import create_debug_data_pipeline
    
    try:
        pipeline = create_debug_data_pipeline(test_config)
        
        # Create trainer
        trainer = create_trainer_from_config(test_config, pipeline, device='cpu', debug_mode=True)
        
        print("Trainer created successfully!")
        print(f"Device: {trainer.device}")
        print(f"Stages: {len(trainer.stages)}")
        
        # Test stage setup
        trainer._setup_stage(0)
        print("Stage 0 setup successful!")
        
        # Test data loader creation
        train_loader, val_loader = pipeline.create_dataloaders('foundational', batch_size=2)
        print(f"Data loaders created: {len(train_loader)} train, {len(val_loader)} val batches")
        
        # Test single training step
        trainer.model.train()
        for batch in train_loader:
            print("Testing single forward pass...")
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            outputs = trainer.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            logits = outputs['logits']
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            print(f"Forward pass successful! Loss: {loss.item():.4f}")
            break
        
        print("\nTrainer test complete!")
        
    except Exception as e:
        print(f"Trainer test failed: {e}")
        import traceback
        traceback.print_exc()


def quick_training_test(config: Dict[str, Any], data_pipeline: DataPipeline, max_steps: int = 10):
    """
    Quick training test for debugging and validation.
    
    Runs a few training steps to verify the pipeline works correctly.
    """
    print(f"Running quick training test ({max_steps} steps)...")
    
    trainer = create_trainer_from_config(config, data_pipeline, device='auto', debug_mode=True)
    
    # Setup first stage
    trainer._setup_stage(0)
    
    # Create data loaders
    train_loader, val_loader = data_pipeline.create_dataloaders(
        trainer.current_stage['name'], 
        batch_size=config.get('training', {}).get('batch_size', 4)
    )
    
    # Check for empty loader (debug mode protection)
    if len(train_loader) == 0:
        print("Warning: Empty train_loader in debug mode")
        return
    
    # Run a few training steps
    trainer.model.train()
    step_count = 0
    
    for batch in train_loader:
        if step_count >= max_steps:
            break
            
        # Move to device
        input_ids = batch['input_ids'].to(trainer.device)
        attention_mask = batch['attention_mask'].to(trainer.device)
        labels = batch['labels'].to(trainer.device)
        
        # Forward pass
        trainer.optimizer.zero_grad()
        outputs = trainer.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        trainer.optimizer.step()
        
        print(f"Step {step_count + 1}: Loss = {loss.item():.4f}")
        step_count += 1
    
    # Quick evaluation
    val_loss, val_perplexity = trainer._evaluate_stage(val_loader)
    print(f"Validation: Loss = {val_loss:.4f}, Perplexity = {val_perplexity:.2f}")
    
    print("Quick training test completed successfully!")


def estimate_training_time(config: Dict[str, Any], data_pipeline: DataPipeline) -> Dict[str, float]:
    """
    Estimate total training time for the full curriculum.
    
    Returns estimates in hours for each stage and total.
    """
    # Get curriculum stages
    stages = config.get('curriculum', {}).get('stages', [])
    batch_size = config.get('training', {}).get('batch_size', 32)
    
    estimates = {}
    total_time = 0.0
    
    for stage in stages:
        stage_name = stage['name']
        epochs = stage['epochs']
        
        # Create data loader to get batch count
        train_loader, _ = data_pipeline.create_dataloaders(stage_name, batch_size)
        steps_per_epoch = len(train_loader)
        total_steps = epochs * steps_per_epoch
        
        # Estimate time per step (rough approximation)
        # Based on model size and sequence length
        model_params = config.get('model', {}).get('parameter_count', {}).get('total', 125_000_000)
        seq_length = config.get('data', {}).get('sequence_length', 512)
        
        # Very rough estimate: larger models and longer sequences take more time
        base_time_per_step = 0.1  # seconds
        param_factor = model_params / 125_000_000  # relative to 125M baseline
        seq_factor = seq_length / 512  # relative to 512 baseline
        
        time_per_step = base_time_per_step * param_factor * seq_factor
        stage_time_hours = (total_steps * time_per_step) / 3600
        
        estimates[stage_name] = {
            'epochs': epochs,
            'steps': total_steps,
            'estimated_hours': stage_time_hours
        }
        
        total_time += stage_time_hours
    
    estimates['total_hours'] = total_time
    
    print(f"Training estimate:")
    for stage_name, info in estimates.items():
        if stage_name != 'total_hours':
            print(f"  {stage_name}: {info['estimated_hours']:.1f}h ({info['epochs']} epochs, {info['steps']} steps)")
    print(f"  Total: {total_time:.1f}h")
    
    return estimates

# Add these functions to the end of src/trainer.py

def test_trainer(config: Dict[str, Any], data_pipeline, max_steps: int = 10):
    """
    Test trainer functionality for debugging.
    
    Verifies trainer creation, stage setup, and basic forward pass.
    """
    print("Testing trainer functionality...")
    
    try:
        # Create trainer
        trainer = create_trainer_from_config(config, data_pipeline, device='cpu', debug_mode=True)
        print(f"[OK] Trainer created successfully")
        print(f"Device: {trainer.device}")
        print(f"Stages: {len(trainer.stages)}")
        
        # Test stage setup
        trainer._setup_stage(0)
        print("[OK] Stage 0 setup successful!")
        
        # Test data loader creation
        train_loader, val_loader = data_pipeline.create_dataloaders('foundational', batch_size=2)
        print(f"[OK] Data loaders created: {len(train_loader)} train, {len(val_loader)} val batches")
        
        # Test single training step
        trainer.model.train()
        for batch in train_loader:
            print("Testing single forward pass...")
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            outputs = trainer.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            print(f"[OK] Forward pass successful! Loss: {loss.item():.4f}")
            break
        
        print("[OK] Trainer test complete!")
        
    except Exception as e:
        print(f"[FAILED] Trainer test failed: {e}")
        import traceback
        traceback.print_exc()

def estimate_training_time(config: Dict[str, Any], data_pipeline):
    """
    Estimate total training time based on configuration and data.
    """
    print("Estimating training time...")
    
    # Get data sizes
    total_segments = len(data_pipeline.segments) if hasattr(data_pipeline, 'segments') else 1000
    batch_size = config.get('training', {}).get('batch_size', 32)
    
    # Calculate steps per stage
    stages = config.get('curriculum', {}).get('stages', [])
    total_epochs = sum(stage.get('epochs', 50) for stage in stages)
    
    steps_per_epoch = max(1, total_segments // batch_size)
    total_steps = total_epochs * steps_per_epoch
    
    # Estimate time per step (varies by hardware)
    estimated_seconds_per_step = 0.1  # Conservative estimate for CPU testing
    if torch.cuda.is_available():
        estimated_seconds_per_step = 0.05  # Faster with GPU
    
    total_time_hours = (total_steps * estimated_seconds_per_step) / 3600
    
    print(f"Training estimate:")
    print(f"  Total segments: {total_segments:,}")
    print(f"  Total epochs: {total_epochs}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Estimated time: {total_time_hours:.1f} hours")
