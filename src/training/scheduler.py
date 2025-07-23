"""
Curriculum Scheduler - FIXED CONFIG INTEGRATION
All hardcoded values moved to configuration
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class CurriculumScheduler:
    """Manages curriculum learning schedule and transitions - FIXED VERSION"""
    
    def __init__(self, curriculum_config, training_config=None):
        self.config = curriculum_config
        self.training_config = training_config
        self.stages = curriculum_config.stages
        
        # FIXED: Get parameters from config instead of hardcoding
        if training_config:
            self.adaptation_window = getattr(training_config, 'adaptation_window', 10)
            self.improvement_threshold = getattr(training_config, 'improvement_threshold', 0.01)
        else:
            # Fallback defaults if no training config provided
            self.adaptation_window = 10
            self.improvement_threshold = 0.01
        
        # Calculate cumulative epoch boundaries
        self.stage_boundaries = []
        cumulative_epochs = 0
        for stage in self.stages:
            cumulative_epochs += stage.epochs
            self.stage_boundaries.append(cumulative_epochs)
    
    def get_current_stage(self, epoch: int) -> Tuple[int, any]:
        """Get current stage index and config for given epoch"""
        for i, boundary in enumerate(self.stage_boundaries):
            if epoch < boundary:
                return i, self.stages[i]
        # Return last stage if epoch exceeds total
        return len(self.stages) - 1, self.stages[-1]
    
    def get_stage_progress(self, epoch: int) -> float:
        """Get progress within current stage (0.0 to 1.0)"""
        stage_idx, stage_config = self.get_current_stage(epoch)
        
        # Calculate epoch within this stage
        start_epoch = 0 if stage_idx == 0 else self.stage_boundaries[stage_idx - 1]
        epoch_in_stage = epoch - start_epoch
        
        progress = epoch_in_stage / stage_config.epochs
        return min(1.0, max(0.0, progress))
    
    def get_masking_rate(self, epoch: int, stage_config=None) -> float:
        """Get masking rate for current epoch"""
        if stage_config is None:
            _, stage_config = self.get_current_stage(epoch)
        
        # Get stage progress
        progress = self.get_stage_progress(epoch)
        
        # Interpolate masking rate based on progress
        min_rate, max_rate = stage_config.masking_rate_range
        
        # Default: linear interpolation from max to min
        # (Start with high masking, gradually reduce)
        masking_rate = max_rate - progress * (max_rate - min_rate)
        
        return masking_rate
    
    def should_transition_stage(self, epoch: int) -> bool:
        """Check if we should transition to next stage"""
        for boundary in self.stage_boundaries[:-1]:  # Exclude last boundary
            if epoch == boundary:
                return True
        return False


class AdaptiveScheduler:
    """Adaptive curriculum scheduler - FIXED CONFIG VERSION"""
    
    def __init__(self, base_scheduler: CurriculumScheduler, 
                 training_config=None):
        self.base_scheduler = base_scheduler
        
        # FIXED: Get all parameters from config
        if training_config:
            self.adaptation_window = getattr(training_config, 'adaptation_window', 10)
            self.improvement_threshold = getattr(training_config, 'improvement_threshold', 0.01)
            self.min_epochs_per_stage = getattr(training_config, 'min_epochs_per_stage', 5)
            self.max_epochs_per_stage = getattr(training_config, 'max_epochs_per_stage', 200)
            self.performance_plateau_epochs = getattr(training_config, 'performance_plateau_epochs', 10)
        else:
            # Fallback defaults
            self.adaptation_window = 10
            self.improvement_threshold = 0.01
            self.min_epochs_per_stage = 5
            self.max_epochs_per_stage = 200
            self.performance_plateau_epochs = 10
        
        # Track performance history
        self.loss_history = []
        self.stage_performance = {}
    
    def update_performance(self, epoch: int, loss: float):
        """Update performance history"""
        self.loss_history.append((epoch, loss))
        
        # Keep only recent history
        if len(self.loss_history) > self.adaptation_window * 2:
            self.loss_history = self.loss_history[-self.adaptation_window:]
    
    def should_extend_stage(self, epoch: int) -> bool:
        """Check if current stage should be extended"""
        if epoch < self.min_epochs_per_stage:
            return False
            
        if len(self.loss_history) < self.adaptation_window:
            return False
        
        # Check recent improvement
        recent_losses = [loss for _, loss in self.loss_history[-self.adaptation_window:]]
        
        if len(recent_losses) < 2:
            return False
        
        # Calculate improvement rate
        start_loss = recent_losses[0]
        end_loss = recent_losses[-1]
        
        if start_loss <= 0:
            return False
        
        improvement = (start_loss - end_loss) / start_loss
        
        # Extend stage if still improving significantly
        return improvement > self.improvement_threshold


# FIXED: Add missing imports and configuration support
class MaskingStrategies:
    """Different masking strategies with configurable parameters"""
    
    def __init__(self, config=None):
        # FIXED: Accept config for strategy parameters
        self.config = config
        self.default_span_length = getattr(config, 'masking_span_length', 3.0) if config else 3.0
        self.difficulty_temperature = getattr(config, 'difficulty_temperature', 1.0) if config else 1.0
    
    @staticmethod
    def uniform_random(input_ids: 'torch.Tensor', masking_rate: float, 
                      mask_token_id: int, pad_token_id: int) -> 'torch.Tensor':
        """Standard uniform random masking"""
        import torch
        
        batch_size, seq_len = input_ids.shape
        
        # Don't mask padding tokens
        valid_mask = (input_ids != pad_token_id)
        
        # Create random mask
        random_mask = torch.rand_like(input_ids, dtype=torch.float) < masking_rate
        
        # Apply only to valid (non-padding) positions
        final_mask = random_mask & valid_mask
        
        # Apply masking
        masked_input = input_ids.clone()
        masked_input[final_mask] = mask_token_id
        
        return masked_input, final_mask
    
    def span_masking(self, input_ids: 'torch.Tensor', masking_rate: float,
                    mask_token_id: int, pad_token_id: int) -> 'torch.Tensor':
        """Mask contiguous spans with configurable length"""
        import torch
        
        batch_size, seq_len = input_ids.shape
        masked_input = input_ids.clone()
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for i in range(batch_size):
            valid_positions = (input_ids[i] != pad_token_id).nonzero(as_tuple=True)[0]
            if len(valid_positions) == 0:
                continue
            
            n_valid = len(valid_positions)
            n_mask = int(n_valid * masking_rate)
            
            # Generate spans with configurable length
            masked_positions = set()
            while len(masked_positions) < n_mask:
                # Sample span start
                start_idx = torch.randint(0, n_valid, (1,)).item()
                start_pos = valid_positions[start_idx].item()
                
                # Sample span length using configured parameter
                span_length = max(1, int(torch.poisson(torch.tensor(self.default_span_length - 1)).item()) + 1)
                
                # Add span positions
                for j in range(span_length):
                    pos = start_pos + j
                    if pos < seq_len and input_ids[i, pos] != pad_token_id:
                        masked_positions.add(pos)
                    
                    if len(masked_positions) >= n_mask:
                        break
            
            # Apply masking
            for pos in masked_positions:
                masked_input[i, pos] = mask_token_id
                mask[i, pos] = True
        
        return masked_input, mask