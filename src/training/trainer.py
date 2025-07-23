"""
Enhanced Curriculum Trainer with Dynamic Adaptation - FIXED VERSION
Implements research-aligned curriculum learning for diffusion models
"""

import os
import time
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from training.scheduler import CurriculumScheduler
from training.metrics import TrainingMetrics
from training.format_datasets import SentenceDataset, PairDataset, ParagraphDataset
from model.diffusion import MaskedDiffusionLM
from data.pipeline import TextSegment, ArgumentMiner


class AdaptiveCurriculumScheduler:
    """Enhanced scheduler with dynamic adaptation"""
    
    def __init__(self, curriculum_config, pipeline, training_config):  # Add training_config parameter
        self.config = curriculum_config
        self.pipeline = pipeline
        self.current_stage = 0
        self.current_vocab_level = 1
        self.stage_performance_history = defaultdict(list)
        self.adaptation_window = training_config.adaptation_window  # Now properly referenced
        self.improvement_threshold = training_config.improvement_threshold
        
        # Stage transition criteria
        self.min_epochs_per_stage = training_config.min_epochs_per_stage
        self.max_epochs_per_stage = training_config.max_epochs_per_stage
        self.performance_plateau_epochs = training_config.performance_plateau_epochs
        
    def should_advance_stage(self, epoch: int, recent_losses: List[float]) -> bool:
        """Determine if we should move to next curriculum stage"""
        if epoch < self.min_epochs_per_stage:
            return False
            
        if len(recent_losses) < self.adaptation_window:
            return False
            
        # Check for performance plateau
        if len(recent_losses) >= self.performance_plateau_epochs:
            recent_avg = np.mean(recent_losses[-5:])
            older_avg = np.mean(recent_losses[-self.performance_plateau_epochs:-5])
            improvement = (older_avg - recent_avg) / max(older_avg, 1e-8)
            
            if improvement < self.improvement_threshold:
                return True
                
        # Force advance after max epochs
        return epoch >= self.max_epochs_per_stage
    
    def should_advance_vocab_level(self, current_loss: float, stage_start_loss: float) -> bool:
        """Determine if vocabulary should be expanded"""
        if stage_start_loss <= 0:
            return False
            
        # Advance when loss has improved by 30%
        improvement = (stage_start_loss - current_loss) / stage_start_loss
        return improvement > 0.3
    
    def get_current_masking_rate(self, epoch_in_stage: int, total_stage_epochs: int) -> float:
        """Get masking rate for current stage and progress"""
        stage_config = self.config.stages[self.current_stage]
        min_rate, max_rate = stage_config.masking_rate_range
        
        # Linear decrease from max to min over the stage
        progress = min(1.0, epoch_in_stage / max(total_stage_epochs, 1))
        return max_rate - progress * (max_rate - min_rate)


class ArgumentPairDataset(Dataset):
    """Dataset for argument pairs (Stage II)"""
    
    def __init__(self, argument_pairs, tokenizer, max_length=512):
        self.pairs = argument_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep_token = tokenizer.sep_token or " [SEP] "
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        evidence_segment, claim_segment = self.pairs[idx]
        
        # Format as evidence [SEP] claim  
        text = f"{evidence_segment.text}{self.sep_token}{claim_segment.text}"
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        # Pad if needed
        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'evidence_difficulty': evidence_segment.combined_difficulty,
            'claim_difficulty': claim_segment.combined_difficulty
        }


class SelfTrainingDataset(Dataset):
    """Dataset with pseudo-generated data for Stage III"""
    
    def __init__(self, original_segments, pseudo_texts, tokenizer, max_length=512):
        self.segments = original_segments
        self.pseudo_texts = pseudo_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.segments) + len(self.pseudo_texts)
    
    def __getitem__(self, idx):
        if idx < len(self.segments):
            text = self.segments[idx].text
            difficulty = self.segments[idx].combined_difficulty
            is_pseudo = False
        else:
            text = self.pseudo_texts[idx - len(self.segments)]
            difficulty = 0.5  # Default difficulty for pseudo data
            is_pseudo = True
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        # Pad if needed
        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'difficulty': difficulty,
            'is_pseudo': is_pseudo
        }


class EnhancedCurriculumTrainer:
    """Main trainer with enhanced curriculum learning"""
    
    def __init__(self,
                 model: MaskedDiffusionLM,
                 pipeline,  # Enhanced TextDataPipeline
                 val_data: List,
                 config,
                 output_dir: str):
        
        self.model = model
        self.pipeline = pipeline
        self.val_data = val_data
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced curriculum components
        self.curriculum_scheduler = AdaptiveCurriculumScheduler(
            config.curriculum, pipeline, config.training
        )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.stage_start_epoch = 0
        self.stage_start_loss = None
        self.best_val_loss = float('inf')
        
        # Dynamic difficulty tracking
        self.batch_difficulties = []
        self.batch_losses = []
        self.batch_gradients = []
        
        # Stage-specific tokenizers
        self.tokenizers = {}
        self.load_stage_tokenizers()
        
        # Setup training components
        self._setup_optimizer()
        self._setup_logging()
        self._setup_mixed_precision()
        
        # Metrics tracking
        self.metrics = TrainingMetrics()
        
        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def load_stage_tokenizers(self):
        """Load tokenizers for different vocabulary levels"""
        # First, try to load vocabulary curriculum tokenizers
        data_dir = Path(getattr(self.config, 'data_dir', 'data/processed'))
        vocab_curriculum_exists = False
        
        # Check if vocabulary curriculum tokenizers exist
        for level in range(1, 6):
            tokenizer_path = data_dir / f"tokenizer_level_{level}"
            if tokenizer_path.exists():
                vocab_curriculum_exists = True
                break
        
        if vocab_curriculum_exists:
            print("🔧 Loading vocabulary curriculum tokenizers...")
            # Original vocabulary curriculum code
            for level in range(1, 6):
                tokenizer_path = data_dir / f"tokenizer_level_{level}"
                if tokenizer_path.exists():
                    from transformers import AutoTokenizer
                    self.tokenizers[level] = AutoTokenizer.from_pretrained(tokenizer_path)
                    print(f"Loaded tokenizer level {level}: {len(self.tokenizers[level]):,} tokens")
                else:
                    # Fall back to standard tokenizer
                    fallback_tokenizer = self.pipeline.tokenizer or self.create_default_tokenizer()
                    self.tokenizers[level] = fallback_tokenizer
                    print(f"Using fallback tokenizer for level {level}: {len(fallback_tokenizer):,} tokens")
        else:
            print("🔧 No vocabulary curriculum found - using single tokenizer for all levels")
            # Use the pipeline's tokenizer (should be the compressed one) for all levels
            main_tokenizer = self.pipeline.tokenizer or self.create_default_tokenizer()
            for level in range(1, 6):
                self.tokenizers[level] = main_tokenizer
            
            print(f"✅ Using single tokenizer for all levels: {len(main_tokenizer):,} tokens")

    def validate_model_vocab_compatibility(self, tokenizer, stage: int, vocab_level: int):
        """Ensure model vocab size matches tokenizer"""
        model_vocab_size = self.model.vocab_size
        tokenizer_vocab_size = len(tokenizer)
        
        if model_vocab_size != tokenizer_vocab_size:
            self.logger.warning(
                f"Stage {stage}, Vocab {vocab_level}: Model vocab size ({model_vocab_size:,}) != "
                f"Tokenizer vocab size ({tokenizer_vocab_size:,})"
            )
            
            # If model is smaller, we need to adjust the model
            if model_vocab_size < tokenizer_vocab_size:
                self.logger.info(f"Expanding model vocabulary from {model_vocab_size} to {tokenizer_vocab_size}")
                self._expand_model_vocabulary(tokenizer_vocab_size)
            else:
                self.logger.warning("Model vocabulary larger than tokenizer - this may cause issues")

    def _expand_model_vocabulary(self, new_vocab_size: int):
        """Expand model vocabulary to match tokenizer"""
        old_vocab_size = self.model.vocab_size
        
        # Expand token embedding layer
        old_embeddings = self.model.transformer.token_embedding.weight.data
        new_embeddings = torch.zeros(new_vocab_size, self.model.d_model, device=old_embeddings.device)
        new_embeddings[:old_vocab_size] = old_embeddings
        
        # Initialize new embeddings with small random values
        if new_vocab_size > old_vocab_size:
            torch.nn.init.normal_(new_embeddings[old_vocab_size:], mean=0.0, std=0.02)
        
        # Replace embedding layer
        self.model.transformer.token_embedding = torch.nn.Embedding(
            new_vocab_size, self.model.d_model, 
            padding_idx=self.model.pad_token_id
        )
        self.model.transformer.token_embedding.weight.data = new_embeddings
        
        # Expand output layer
        old_lm_head = self.model.transformer.lm_head.weight.data
        new_lm_head = torch.zeros(new_vocab_size, self.model.d_model, device=old_lm_head.device)
        new_lm_head[:old_vocab_size] = old_lm_head
        
        if new_vocab_size > old_vocab_size:
            torch.nn.init.normal_(new_lm_head[old_vocab_size:], mean=0.0, std=0.02)
        
        self.model.transformer.lm_head = torch.nn.Linear(
            self.model.d_model, new_vocab_size, bias=False
        )
        self.model.transformer.lm_head.weight.data = new_lm_head
        
        # Update vocab size
        self.model.vocab_size = new_vocab_size
        
        # Move to device
        self.model.to(self.device)
        
        # Reinitialize optimizer to include new parameters
        self._setup_optimizer()

    def create_stage_dataloader(self, stage: int, vocab_level: int, batch_size: int) -> DataLoader:
        """Create curriculum-aware dataloader for specific stage with fallback"""
        
        # Use the appropriate tokenizer for this vocabulary level
        tokenizer = self.tokenizers.get(vocab_level, self.tokenizers[1])
        
        # Validate model-tokenizer compatibility (only if vocab curriculum is active)
        if hasattr(self, 'vocab_curriculum_active') and self.vocab_curriculum_active:
            self.validate_model_vocab_compatibility(tokenizer, stage, vocab_level)
        
        # Rest of the method stays the same...
        if stage == 1:
            # Foundation: Individual sentences
            segments = self.get_stage_data_with_fallback(stage, vocab_level)
            dataset = SentenceDataset(segments, tokenizer, self.config.model.max_seq_len)
            
        elif stage == 2:
            # Structural: Argument pairs
            pairs = self.pipeline.get_argument_pairs(stage)
            
            # Filter by vocabulary level with fallback
            filtered_pairs = []
            for evidence, claim in pairs:
                if (evidence.vocabulary_level <= vocab_level and 
                    claim.vocabulary_level <= vocab_level):
                    filtered_pairs.append((evidence, claim))
            
            # Fallback: If no pairs found, create pairs from available segments
            if len(filtered_pairs) == 0:
                segments = self.get_stage_data_with_fallback(stage, vocab_level)
                self.logger.warning(f"No argument pairs found. Creating pairs from {len(segments)} segments")
                
                # Create simple pairs from consecutive segments
                for i in range(0, len(segments) - 1, 2):
                    if i + 1 < len(segments):
                        filtered_pairs.append((segments[i], segments[i + 1]))
                
                # If still no pairs, duplicate segments
                if len(filtered_pairs) == 0 and segments:
                    filtered_pairs = [(segments[0], segments[0])]
            
            dataset = ArgumentPairDataset(filtered_pairs, tokenizer, self.config.model.max_seq_len)
            
        elif stage == 3:
            # Refinement: Full paragraphs + pseudo data
            segments = self.get_stage_data_with_fallback(stage, vocab_level)
            
            # Generate pseudo data if we have a trained model
            pseudo_texts = []
            if hasattr(self, 'stage_2_model_path') and os.path.exists(self.stage_2_model_path):
                stage_2_segments = self.get_stage_data_with_fallback(2, vocab_level)
                pseudo_texts = self.pipeline.generate_pseudo_data(
                    None, stage_2_segments, num_samples=min(
                        getattr(self.config.curriculum, 'pseudo_data_max_samples', 100),
                        int(len(segments) * getattr(self.config.curriculum, 'pseudo_data_ratio', 0.25))
                    )
                )
            
            dataset = SelfTrainingDataset(segments, pseudo_texts, tokenizer, self.config.model.max_seq_len)
            
        else:
            raise ValueError(f"Invalid stage: {stage}")
        
        # Ensure dataset is not empty
        if len(dataset) == 0:
            raise ValueError(f"Dataset is empty for stage {stage}, vocab level {vocab_level}")
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=self.config.training.pin_memory,
            drop_last=True if len(dataset) > batch_size else False  # Don't drop last if dataset is small
        )
    
    def create_default_tokenizer(self):
        """Create default tokenizer if none available"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        special_tokens = {
            "pad_token": "<pad>",
            "mask_token": "<mask>",
            "bos_token": "<bos>",
            "eos_token": "<eos>"
        }
        
        tokens_to_add = {}
        for key, token in special_tokens.items():
            if getattr(tokenizer, key) is None:
                tokens_to_add[key] = token
        
        if tokens_to_add:
            tokenizer.add_special_tokens(tokens_to_add)
        
        return tokenizer
    
    def _setup_optimizer(self):
        """Initialize optimizer with curriculum-aware scheduling"""
        # Parameter groups with different decay rates
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
        
        # Dynamic learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.training.scheduler_restart_period,  # Will be adjusted per stage
            T_mult=2,
            eta_min=self.config.training.min_learning_rate
        )
        
    def _setup_logging(self):
        """Setup comprehensive logging with proper file handle management"""
        # Create a named logger instead of using basicConfig
        self.logger = logging.getLogger(f"trainer_{id(self)}")  # Unique logger name
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        log_file = self.output_dir / 'enhanced_train.log'
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)
        
        # Console handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        self.console_handler.setFormatter(formatter)
        self.logger.addHandler(self.console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
        
        # Enhanced tensorboard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
    def cleanup_logging(self):
        """Properly cleanup all logging resources"""
        # Close and remove file handler
        if hasattr(self, 'file_handler'):
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)
            del self.file_handler
        
        # Close and remove console handler
        if hasattr(self, 'console_handler'):
            self.console_handler.close()
            self.logger.removeHandler(self.console_handler)
            del self.console_handler
        
        # Close tensorboard writer
        if hasattr(self, 'writer'):
            self.writer.close()
            del self.writer
        
        # Clear logger handlers
        self.logger.handlers.clear()
        
    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        self.use_amp = self.config.training.use_mixed_precision and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
    
    def get_stage_data_with_fallback(self, stage: int, vocab_level: int) -> List[TextSegment]:
        """Get stage data with progressive fallback to ensure non-empty dataset"""
        # Try exact filtering first
        segments = [s for s in self.pipeline.segments 
                   if s.stage_assignment == stage and s.vocabulary_level <= vocab_level]
        
        if len(segments) > 0:
            self.logger.info(f"Found {len(segments)} segments for stage {stage}, vocab level {vocab_level}")
            return segments
        
        self.logger.warning(f"No segments found for stage {stage}, vocab level {vocab_level}. Trying fallbacks...")
        
        # Fallback 1: Ignore vocabulary level restriction
        segments = [s for s in self.pipeline.segments if s.stage_assignment == stage]
        if len(segments) > 0:
            self.logger.info(f"Fallback 1: Found {len(segments)} segments for stage {stage} (ignoring vocab level)")
            return segments
        
        # Fallback 2: Use all segments if stage filtering fails
        segments = self.pipeline.segments
        self.logger.warning(f"Fallback 2: Using all {len(segments)} segments (ignoring stage and vocab filtering)")
        
        # If still empty, create minimal test data
        if len(segments) == 0:
            self.logger.error("No segments available! Creating minimal test data...")
            test_segment = TextSegment(
                text="This is a test sentence for training.",
                index=0,
                lexical_rarity=0.5,
                syntactic_complexity=0.5,
                thematic_centrality=0.5,
                combined_difficulty=0.5,
                stage_assignment=stage,
                vocabulary_level=vocab_level,
                length=7
            )
            segments = [test_segment]
        
        return segments
    
    def create_stage_dataloader(self, stage: int, vocab_level: int, batch_size: int) -> DataLoader:
        """Create curriculum-aware dataloader for specific stage with fallback"""
        tokenizer = self.tokenizers.get(vocab_level, self.tokenizers[1])
        
        if stage == 1:
            # Foundation: Individual sentences
            segments = self.get_stage_data_with_fallback(stage, vocab_level)
            dataset = SentenceDataset(segments, tokenizer, self.config.model.max_seq_len)
            
        elif stage == 2:
            # Structural: Argument pairs
            pairs = self.pipeline.get_argument_pairs(stage)
            
            # Filter by vocabulary level with fallback
            filtered_pairs = []
            for evidence, claim in pairs:
                if (evidence.vocabulary_level <= vocab_level and 
                    claim.vocabulary_level <= vocab_level):
                    filtered_pairs.append((evidence, claim))
            
            # Fallback: If no pairs found, create pairs from available segments
            if len(filtered_pairs) == 0:
                segments = self.get_stage_data_with_fallback(stage, vocab_level)
                self.logger.warning(f"No argument pairs found. Creating pairs from {len(segments)} segments")
                
                # Create simple pairs from consecutive segments
                for i in range(0, len(segments) - 1, 2):
                    if i + 1 < len(segments):
                        filtered_pairs.append((segments[i], segments[i + 1]))
                
                # If still no pairs, duplicate segments
                if len(filtered_pairs) == 0 and segments:
                    filtered_pairs = [(segments[0], segments[0])]
            
            dataset = ArgumentPairDataset(filtered_pairs, tokenizer, self.config.model.max_seq_len)
            
        elif stage == 3:
            # Refinement: Full paragraphs + pseudo data
            segments = self.get_stage_data_with_fallback(stage, vocab_level)
            
            # Generate pseudo data if we have a trained model
            pseudo_texts = []
            if hasattr(self, 'stage_2_model_path') and os.path.exists(self.stage_2_model_path):
                stage_2_segments = self.get_stage_data_with_fallback(2, vocab_level)
                pseudo_texts = self.pipeline.generate_pseudo_data(
                    None, stage_2_segments, num_samples=min(
                        getattr(self.config.curriculum, 'pseudo_data_max_samples', 100),
                        int(len(segments) * getattr(self.config.curriculum, 'pseudo_data_ratio', 0.25))
                    )
                )
            
            dataset = SelfTrainingDataset(segments, pseudo_texts, tokenizer, self.config.model.max_seq_len)
            
        else:
            raise ValueError(f"Invalid stage: {stage}")
        
        # Ensure dataset is not empty
        if len(dataset) == 0:
            raise ValueError(f"Dataset is empty for stage {stage}, vocab level {vocab_level}")
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=self.config.training.pin_memory,
            drop_last=True if len(dataset) > batch_size else False  # Don't drop last if dataset is small
        )
    
    def train_epoch(self, dataloader: DataLoader, stage: int, vocab_level: int) -> Dict[str, float]:
        """Train one epoch with dynamic difficulty updates"""
        self.model.train()
        epoch_metrics = {'loss': 0.0, 'num_batches': 0}
        
        # Track for dynamic updates
        batch_segment_ids = []
        batch_losses = []
        batch_grad_norms = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Get dynamic masking rate
            masking_rate = self.curriculum_scheduler.get_current_masking_rate(
                self.current_epoch - self.stage_start_epoch,
                self.config.curriculum.stages[stage-1].epochs
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
            
            # Collect gradient norms for dynamic difficulty
            if batch_idx % self.config.training.gradient_accumulation_steps == 0:
                total_grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                
                # Update weights
                if self.use_amp:
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
                
                # Store for dynamic updates
                batch_grad_norms.append(total_grad_norm)
            
            # Track metrics
            epoch_metrics['loss'] += loss.item() * self.config.training.gradient_accumulation_steps
            epoch_metrics['num_batches'] += 1
            batch_losses.append(loss.item() * self.config.training.gradient_accumulation_steps)
            
            # Log progress
            if self.global_step % self.config.training.log_every == 0:
                lr = self.lr_scheduler.get_last_lr()[0]
                self.logger.info(
                    f"Stage {stage}, Vocab {vocab_level}, Step {self.global_step}: "
                    f"Loss={loss.item():.4f}, LR={lr:.2e}, Mask_Rate={masking_rate:.2f}"
                )
                
                # Tensorboard logging
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', lr, self.global_step)
                self.writer.add_scalar('train/masking_rate', masking_rate, self.global_step)
                self.writer.add_scalar('train/stage', stage, self.global_step)
                self.writer.add_scalar('train/vocab_level', vocab_level, self.global_step)
        
        # Update dynamic difficulty scores
        if hasattr(self, 'pipeline') and len(batch_losses) > 0:
            # Only update when we have gradient norms (every gradient_accumulation_steps)
            if batch_idx % self.config.training.gradient_accumulation_steps == 0 and len(batch_grad_norms) > 0:
                # Use the most recent loss and gradient norm
                recent_loss = batch_losses[-1]
                recent_grad_norm = batch_grad_norms[-1]
                
                # Sample one segment to update
                if len(self.pipeline.segments) > 0:
                    sample_segment = random.randint(0, len(self.pipeline.segments) - 1)
                    
                    self.pipeline.update_dynamic_scores(
                        [sample_segment],
                        [recent_loss],
                        [recent_grad_norm]
                    )
        
        # Average metrics
        if epoch_metrics['num_batches'] > 0:
            epoch_metrics['loss'] /= epoch_metrics['num_batches']
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation with current model"""
        self.model.eval()
        
        # Use Stage 1 format for validation
        tokenizer = self.tokenizers[1]
        val_dataset = SentenceDataset(self.val_data, tokenizer, self.config.model.max_seq_len)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.training.val_batch_size_actual,
            shuffle=False,
            num_workers=2
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
                masking_rate=self.config.training.validation_masking_rate
            )
            loss = self.model.compute_loss(outputs)
            
            valid_tokens = attention_mask.sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            num_batches += 1
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        self.model.train()
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'num_batches': num_batches
        }
    
    def save_checkpoint(self, stage: int, vocab_level: int, is_best: bool = False):
        """Save enhanced checkpoint with curriculum state"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'current_stage': stage,
            'current_vocab_level': vocab_level,
            'stage_start_epoch': self.stage_start_epoch,
            'stage_start_loss': self.stage_start_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'curriculum_state': {
                'stage_performance_history': dict(self.curriculum_scheduler.stage_performance_history),
                'current_stage': self.curriculum_scheduler.current_stage,
                'current_vocab_level': self.curriculum_scheduler.current_vocab_level
            }
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save stage-specific checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_stage_{stage}_vocab_{vocab_level}_step_{self.global_step}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved: stage {stage}, vocab {vocab_level}, loss {self.best_val_loss:.4f}")
        
        # Save stage completion checkpoint
        if stage == 2:  # Save Stage 2 model for pseudo-data generation
            self.stage_2_model_path = self.output_dir / 'stage_2_model.pt'
            torch.save(checkpoint, self.stage_2_model_path)
    
    def train(self):
        """Enhanced curriculum training with dynamic adaptation"""
        self.logger.info("Starting enhanced curriculum training...")
        
        recent_losses = deque(maxlen=self.config.training.loss_history_window)
        
        try:
            # Training loop through curriculum stages
            for stage_idx in range(len(self.config.curriculum.stages)):
                stage = stage_idx + 1
                stage_config = self.config.curriculum.stages[stage_idx]
                
                self.curriculum_scheduler.current_stage = stage_idx
                self.stage_start_epoch = self.current_epoch
                self.stage_start_loss = None
                
                self.logger.info(f"\n{'='*20} STAGE {stage}: {stage_config.name.upper()} {'='*20}")
                
                # Vocabulary curriculum within each stage
                max_vocab_level = getattr(self.pipeline.vocab_curriculum, 'max_level', 3) if self.pipeline.vocab_curriculum else 1
                
                # Share epoch budget across all vocab levels in this stage
                total_stage_epochs = 0
                
                for vocab_level in range(1, max_vocab_level + 1):
                    if vocab_level > 1:
                        self.logger.info(f"\n--- Advancing to Vocabulary Level {vocab_level} ---")
                    
                    self.curriculum_scheduler.current_vocab_level = vocab_level
                    
                    # Create stage and vocab-specific dataloader
                    try:
                        train_dataloader = self.create_stage_dataloader(
                            stage, vocab_level, self.config.training.batch_size
                        )
                    except ValueError as e:
                        self.logger.error(f"Failed to create dataloader: {e}")
                        continue
                    
                    if len(train_dataloader) == 0:
                        self.logger.warning(f"No data for stage {stage}, vocab level {vocab_level}")
                        continue
                    
                    # Reset optimizer for new stage
                    if stage > 1 and vocab_level == 1 and self.config.curriculum.reset_optimizer:
                        self._setup_optimizer()
                        self.logger.info("Optimizer reset for new stage")
                    
                    # Train for this vocab level (sharing stage epoch budget)
                    vocab_start_loss = None
                    
                    while total_stage_epochs < stage_config.epochs:
                        self.current_epoch += 1
                        total_stage_epochs += 1
                        
                        # Train epoch
                        start_time = time.time()
                        train_metrics = self.train_epoch(train_dataloader, stage, vocab_level)
                        epoch_time = time.time() - start_time
                        
                        if vocab_start_loss is None:
                            vocab_start_loss = train_metrics['loss']
                        if self.stage_start_loss is None:
                            self.stage_start_loss = train_metrics['loss']
                        
                        recent_losses.append(train_metrics['loss'])
                        
                        # Validate periodically
                        if self.current_epoch % self.config.training.validation_frequency == 0:
                            val_metrics = self.validate()
                            
                            # Check for best model
                            if val_metrics['loss'] < self.best_val_loss:
                                self.best_val_loss = val_metrics['loss']
                                self.save_checkpoint(stage, vocab_level, is_best=True)
                            
                            self.logger.info(
                                f"Stage {stage}, Vocab {vocab_level}, Epoch {self.current_epoch}: "
                                f"Train Loss={train_metrics['loss']:.4f}, "
                                f"Val Loss={val_metrics['loss']:.4f}, "
                                f"Val PPL={val_metrics['perplexity']:.2f}, "
                                f"Time={epoch_time:.1f}s"
                            )
                            
                            # Tensorboard logging
                            self.writer.add_scalar('val/loss', val_metrics['loss'], self.global_step)
                            self.writer.add_scalar('val/perplexity', val_metrics['perplexity'], self.global_step)
                        
                        # Check vocabulary advancement
                        if (vocab_level < max_vocab_level and 
                            self.curriculum_scheduler.should_advance_vocab_level(
                                train_metrics['loss'], vocab_start_loss)):
                            self.logger.info(f"Advancing from vocab level {vocab_level} to {vocab_level + 1}")
                            break
                        
                        # Check stage advancement  
                        if self.curriculum_scheduler.should_advance_stage(
                            total_stage_epochs, list(recent_losses)):
                            self.logger.info(f"Stage {stage} completed after {total_stage_epochs} epochs")
                            break
                        
                        # Save checkpoint
                        if self.global_step % self.config.training.save_every == 0:
                            self.save_checkpoint(stage, vocab_level)
                    
                    # Break out of vocab level loop if stage is complete
                    if total_stage_epochs >= stage_config.epochs:
                        break
                    
                    # Record stage performance
                    if recent_losses:
                        self.curriculum_scheduler.stage_performance_history[stage].append(
                            np.mean(list(recent_losses)[-10:])
                        )
                
                self.logger.info(f"Stage {stage} completed")
            
            self.logger.info("Enhanced curriculum training completed!")
            
        finally:
            # Always cleanup logging resources
            self.cleanup_logging()
        
        # Save final curriculum state
        curriculum_state = {
            'stage_performance_history': dict(self.curriculum_scheduler.stage_performance_history),
            'final_stage': self.curriculum_scheduler.current_stage,
            'final_vocab_level': self.curriculum_scheduler.current_vocab_level,
            'pipeline_state': self.pipeline.dynamic_scorer.__dict__
        }
        
        with open(self.output_dir / 'curriculum_final_state.pkl', 'wb') as f:
            import pickle
            pickle.dump(curriculum_state, f)
        
        print(f"Final curriculum state saved to {self.output_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint with curriculum state"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.stage_start_epoch = checkpoint.get('stage_start_epoch', 0)
        self.stage_start_loss = checkpoint.get('stage_start_loss')
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Restore curriculum state
        if 'curriculum_state' in checkpoint:
            curriculum_state = checkpoint['curriculum_state']
            self.curriculum_scheduler.stage_performance_history = defaultdict(
                list, curriculum_state.get('stage_performance_history', {})
            )
            self.curriculum_scheduler.current_stage = curriculum_state.get('current_stage', 0)
            self.curriculum_scheduler.current_vocab_level = curriculum_state.get('current_vocab_level', 1)
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(
            f"Checkpoint loaded: step {self.global_step}, epoch {self.current_epoch}, "
            f"stage {self.curriculum_scheduler.current_stage}, "
            f"vocab level {self.curriculum_scheduler.current_vocab_level}"
        )