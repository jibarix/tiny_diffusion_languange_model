"""
Main Training Loop with Curriculum Learning
Handles multi-stage training with curriculum scheduling
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from .scheduler import CurriculumScheduler
from .metrics import TrainingMetrics
from .format_datasets import SentenceDataset, PairDataset, ParagraphDataset
from ..model.diffusion import MaskedDiffusionLM


class CurriculumTrainer:
    """Main trainer with curriculum learning support"""
    
    def __init__(self,
                 model: MaskedDiffusionLM,
                 curriculum_scheduler: CurriculumScheduler,
                 train_data: Dict[str, List],  # curriculum splits
                 val_data: List,
                 tokenizer,
                 config,
                 output_dir: str):
        
        self.model = model
        self.curriculum_scheduler = curriculum_scheduler
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.current_stage = 0
        self.best_val_loss = float('inf')
        
        # Setup training components
        self._setup_optimizer()
        self._setup_logging()
        self._setup_mixed_precision()
        
        # Metrics tracking
        self.metrics = TrainingMetrics()
        
        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _setup_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        # Separate weight decay for different parameter groups
        no_decay = ['bias', 'LayerNorm.weight', 'norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.training.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate,
            betas=(self.config.training.beta1, self.config.training.beta2),
            eps=self.config.training.eps
        )
        
        # Learning rate scheduler
        total_steps = self.config.curriculum.total_epochs * len(self.train_data['all']) // self.config.training.effective_batch_size
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.training.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
    def _setup_logging(self):
        """Setup logging and tensorboard"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'train.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Tensorboard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        self.use_amp = self.config.training.use_mixed_precision and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            
    def create_dataloader(self, segments: List, stage_config, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Create format-aware dataloader for given stage"""
        
        # Choose dataset class based on format type
        if stage_config.format_type == "sentences":
            dataset = SentenceDataset(segments, self.tokenizer, self.config.model.max_seq_len)
        elif stage_config.format_type == "pairs":
            dataset = PairDataset(segments, self.tokenizer, self.config.model.max_seq_len)
        elif stage_config.format_type == "paragraphs":
            dataset = ParagraphDataset(segments, self.tokenizer, self.config.model.max_seq_len)
        else:
            # Default to sentences
            dataset = SentenceDataset(segments, self.tokenizer, self.config.model.max_seq_len)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=self.config.training.pin_memory,
            drop_last=True
        )
    
    def train_epoch(self, dataloader: DataLoader, stage_config) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        epoch_metrics = {'loss': 0.0, 'num_batches': 0}
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Get masking rate for this stage
            masking_rate = self.curriculum_scheduler.get_masking_rate(
                self.current_epoch, stage_config
            )
            
            # Zero gradients
            if batch_idx % self.config.training.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    masking_rate=masking_rate
                )
                loss = self.model.compute_loss(outputs)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.training.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Unscale gradients for clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.grad_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.grad_clip_norm
                    )
                    self.optimizer.step()
                
                self.lr_scheduler.step()
                self.global_step += 1
            
            # Track metrics
            epoch_metrics['loss'] += loss.item() * self.config.training.gradient_accumulation_steps
            epoch_metrics['num_batches'] += 1
            
            # Logging
            if self.global_step % self.config.training.log_every == 0:
                lr = self.lr_scheduler.get_last_lr()[0]
                self.logger.info(
                    f"Step {self.global_step}: Loss={loss.item():.4f}, "
                    f"LR={lr:.2e}, Mask_Rate={masking_rate:.2f}, "
                    f"Format={stage_config.format_type}"
                )
                
                # Tensorboard logging
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', lr, self.global_step)
                self.writer.add_scalar('train/masking_rate', masking_rate, self.global_step)
        
        # Average metrics
        if epoch_metrics['num_batches'] > 0:
            epoch_metrics['loss'] /= epoch_metrics['num_batches']
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()
        
        # Create validation dataloader using sentences format
        from config.curriculum_config import StageConfig
        sentence_stage = StageConfig(
            name="validation",
            epochs=1,
            masking_rate_range=(0.15, 0.15),
            data_selection="all",
            format_type="sentences"
        )
        
        val_dataloader = self.create_dataloader(
            self.val_data, 
            sentence_stage,
            self.config.training.val_batch_size_actual,
            shuffle=False
        )
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Fixed masking rate for validation
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                masking_rate=0.15  # Standard BERT masking rate
            )
            loss = self.model.compute_loss(outputs)
            
            # Calculate perplexity
            valid_tokens = attention_mask.sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            num_batches += 1
        
        # Calculate metrics
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        self.model.train()
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'num_batches': num_batches
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'current_stage': self.current_stage,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_step_{self.global_step}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
        
        # Keep only recent checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints, keeping only the most recent"""
        checkpoints = list(self.output_dir.glob('checkpoint_step_*.pt'))
        if len(checkpoints) > keep_last:
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            for old_checkpoint in checkpoints[:-keep_last]:
                old_checkpoint.unlink()
    
    def train(self):
        """Main training loop with curriculum stages"""
        self.logger.info("Starting curriculum training...")
        self.logger.info(f"Total stages: {len(self.config.curriculum.stages)}")
        self.logger.info(f"Total epochs: {self.config.curriculum.total_epochs}")
        
        # Training loop over curriculum stages
        for stage_idx, stage_config in enumerate(self.config.curriculum.stages):
            self.current_stage = stage_idx
            self.logger.info(f"\n=== Stage {stage_idx + 1}: {stage_config.name} ===")
            self.logger.info(f"Format: {stage_config.format_type}")
            
            # Get data for this stage
            stage_data = self.train_data[stage_config.data_selection]
            self.logger.info(f"Stage data: {len(stage_data)} examples")
            
            # Create format-specific dataloader
            train_dataloader = self.create_dataloader(
                stage_data, 
                stage_config,  # Pass stage config for format
                self.config.training.batch_size,
                shuffle=True
            )
            
            # Reset optimizer if configured
            if self.config.curriculum.reset_optimizer and stage_idx > 0:
                self._setup_optimizer()
                self.logger.info("Optimizer reset for new stage")
            
            # Train for this stage
            stage_start_epoch = self.current_epoch
            for epoch in range(stage_config.epochs):
                self.current_epoch = stage_start_epoch + epoch
                
                self.logger.info(
                    f"Stage {stage_idx + 1}, Epoch {epoch + 1}/{stage_config.epochs} "
                    f"(Global Epoch {self.current_epoch + 1})"
                )
                
                # Train epoch
                start_time = time.time()
                train_metrics = self.train_epoch(train_dataloader, stage_config)
                epoch_time = time.time() - start_time
                
                # Validate
                if (self.current_epoch + 1) % 5 == 0:  # Validate every 5 epochs
                    val_metrics = self.validate()
                    
                    # Check for best model
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.save_checkpoint(is_best=True)
                    
                    self.logger.info(
                        f"Epoch {self.current_epoch + 1}: "
                        f"Train Loss={train_metrics['loss']:.4f}, "
                        f"Val Loss={val_metrics['loss']:.4f}, "
                        f"Val PPL={val_metrics['perplexity']:.2f}, "
                        f"Time={epoch_time:.1f}s"
                    )
                    
                    # Tensorboard logging
                    self.writer.add_scalar('val/loss', val_metrics['loss'], self.global_step)
                    self.writer.add_scalar('val/perplexity', val_metrics['perplexity'], self.global_step)
                
                # Save checkpoint
                if self.global_step % self.config.training.save_every == 0:
                    self.save_checkpoint()
            
            self.logger.info(f"Stage {stage_idx + 1} completed")
        
        self.logger.info("Training completed!")
        self.writer.close()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.current_stage = checkpoint['current_stage']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: step {self.global_step}, epoch {self.current_epoch}")