"""
Curriculum Scheduler
Handles stage transitions and masking rate scheduling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class CurriculumScheduler:
    """Manages curriculum learning schedule and transitions"""
    
    def __init__(self, curriculum_config):
        self.config = curriculum_config
        self.stages = curriculum_config.stages
        
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
    
    def get_transition_schedule(self, from_stage: int, to_stage: int, 
                             transition_epochs: int) -> List[float]:
        """Get gradual transition schedule between stages"""
        if transition_epochs <= 1:
            return [1.0]  # Immediate transition
        
        # Linear transition
        weights = np.linspace(0.0, 1.0, transition_epochs)
        return weights.tolist()
    
    def get_stage_summary(self) -> Dict:
        """Get summary of curriculum stages"""
        summary = {
            'total_stages': len(self.stages),
            'total_epochs': sum(stage.epochs for stage in self.stages),
            'stages': []
        }
        
        cumulative = 0
        for i, stage in enumerate(self.stages):
            stage_info = {
                'index': i,
                'name': stage.name,
                'epochs': stage.epochs,
                'epoch_range': (cumulative, cumulative + stage.epochs),
                'masking_range': stage.masking_rate_range,
                'data_selection': stage.data_selection,
                'format_type': stage.format_type
            }
            summary['stages'].append(stage_info)
            cumulative += stage.epochs
        
        return summary


class AdaptiveScheduler:
    """Adaptive curriculum scheduler based on model performance"""
    
    def __init__(self, base_scheduler: CurriculumScheduler, 
                 adaptation_window: int = 10,
                 improvement_threshold: float = 0.01):
        self.base_scheduler = base_scheduler
        self.adaptation_window = adaptation_window
        self.improvement_threshold = improvement_threshold
        
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
    
    def get_adapted_masking_rate(self, epoch: int, base_rate: float, 
                                performance_trend: str = "stable") -> float:
        """Adapt masking rate based on performance"""
        
        if performance_trend == "improving":
            # Slightly increase difficulty (lower masking rate)
            return max(0.05, base_rate * 0.95)
        elif performance_trend == "struggling":
            # Make it easier (higher masking rate)
            return min(0.95, base_rate * 1.05)
        else:
            # Keep base rate
            return base_rate
    
    def analyze_performance_trend(self, window: int = 5) -> str:
        """Analyze recent performance trend"""
        if len(self.loss_history) < window:
            return "insufficient_data"
        
        recent_losses = [loss for _, loss in self.loss_history[-window:]]
        
        # Simple trend analysis
        if len(recent_losses) < 2:
            return "stable"
        
        # Calculate slope
        x = np.arange(len(recent_losses))
        y = np.array(recent_losses)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            
            if slope < -0.01:  # Decreasing loss (improving)
                return "improving"
            elif slope > 0.01:  # Increasing loss (struggling)
                return "struggling"
        
        return "stable"


class MaskingStrategies:
    """Different masking strategies for diffusion training"""
    
    @staticmethod
    def uniform_random(input_ids: 'torch.Tensor', masking_rate: float, 
                      mask_token_id: int, pad_token_id: int) -> 'torch.Tensor':
        """Standard uniform random masking"""
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
    
    @staticmethod
    def span_masking(input_ids: 'torch.Tensor', masking_rate: float,
                    mask_token_id: int, pad_token_id: int,
                    span_length_mean: float = 3.0) -> 'torch.Tensor':
        """Mask contiguous spans (like SpanBERT)"""
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
            
            # Generate spans
            masked_positions = set()
            while len(masked_positions) < n_mask:
                # Sample span start
                start_idx = torch.randint(0, n_valid, (1,)).item()
                start_pos = valid_positions[start_idx].item()
                
                # Sample span length (geometric distribution)
                span_length = max(1, int(torch.poisson(torch.tensor(span_length_mean - 1)).item()) + 1)
                
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
    
    @staticmethod
    def difficulty_aware_masking(input_ids: 'torch.Tensor', masking_rate: float,
                               mask_token_id: int, pad_token_id: int,
                               token_difficulties: Optional['torch.Tensor'] = None) -> 'torch.Tensor':
        """Mask tokens based on difficulty scores"""
        import torch
        
        if token_difficulties is None:
            # Fall back to uniform random
            return MaskingStrategies.uniform_random(
                input_ids, masking_rate, mask_token_id, pad_token_id
            )
        
        batch_size, seq_len = input_ids.shape
        masked_input = input_ids.clone()
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for i in range(batch_size):
            valid_positions = (input_ids[i] != pad_token_id).nonzero(as_tuple=True)[0]
            if len(valid_positions) == 0:
                continue
            
            n_valid = len(valid_positions)
            n_mask = int(n_valid * masking_rate)
            
            # Get difficulties for valid positions
            valid_difficulties = token_difficulties[i, valid_positions]
            
            # Sample based on difficulty (higher difficulty = higher probability)
            probabilities = torch.softmax(valid_difficulties, dim=0)
            
            # Sample positions without replacement
            sampled_indices = torch.multinomial(probabilities, n_mask, replacement=False)
            sampled_positions = valid_positions[sampled_indices]
            
            # Apply masking
            masked_input[i, sampled_positions] = mask_token_id
            mask[i, sampled_positions] = True
        
        return masked_input, mask