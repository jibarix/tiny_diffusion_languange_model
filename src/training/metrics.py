"""
Training Metrics
Loss tracking, perplexity calculation, and training progress monitoring
"""

import math
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import numpy as np


class TrainingMetrics:
    """Tracks training metrics across epochs and stages"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Raw metrics storage
        self.step_metrics = []
        self.epoch_metrics = []
        self.stage_metrics = []
        
        # Rolling windows for smooth metrics
        self.loss_window = deque(maxlen=window_size)
        self.lr_window = deque(maxlen=window_size)
        
        # Timing
        self.start_time = time.time()
        self.stage_start_times = {}
        
        # Best metrics tracking
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_val_perplexity = float('inf')
        
    def update_step(self, step: int, loss: float, lr: float, 
                   masking_rate: float = None, **kwargs):
        """Update step-level metrics"""
        timestamp = time.time()
        
        step_data = {
            'step': step,
            'loss': loss,
            'lr': lr,
            'masking_rate': masking_rate,
            'timestamp': timestamp,
            **kwargs
        }
        
        self.step_metrics.append(step_data)
        self.loss_window.append(loss)
        self.lr_window.append(lr)
        
        # Update best training loss
        if loss < self.best_train_loss:
            self.best_train_loss = loss
    
    def update_epoch(self, epoch: int, stage: int, train_loss: float,
                    val_loss: float = None, val_perplexity: float = None,
                    **kwargs):
        """Update epoch-level metrics"""
        timestamp = time.time()
        
        epoch_data = {
            'epoch': epoch,
            'stage': stage,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
            'timestamp': timestamp,
            **kwargs
        }
        
        self.epoch_metrics.append(epoch_data)
        
        # Update best validation metrics
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
        if val_perplexity is not None and val_perplexity < self.best_val_perplexity:
            self.best_val_perplexity = val_perplexity
    
    def start_stage(self, stage_idx: int, stage_name: str):
        """Mark the start of a training stage"""
        self.stage_start_times[stage_idx] = time.time()
        
    def end_stage(self, stage_idx: int, stage_name: str, **kwargs):
        """Mark the end of a training stage"""
        end_time = time.time()
        start_time = self.stage_start_times.get(stage_idx, end_time)
        
        stage_data = {
            'stage_idx': stage_idx,
            'stage_name': stage_name,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            **kwargs
        }
        
        self.stage_metrics.append(stage_data)
    
    def get_smoothed_loss(self) -> float:
        """Get smoothed loss over recent window"""
        if not self.loss_window:
            return 0.0
        return np.mean(list(self.loss_window))
    
    def get_smoothed_lr(self) -> float:
        """Get smoothed learning rate"""
        if not self.lr_window:
            return 0.0
        return np.mean(list(self.lr_window))
    
    def get_training_speed(self, recent_steps: int = 100) -> Dict[str, float]:
        """Calculate training speed metrics"""
        if len(self.step_metrics) < 2:
            return {'steps_per_sec': 0.0, 'tokens_per_sec': 0.0}
        
        recent_metrics = self.step_metrics[-recent_steps:]
        if len(recent_metrics) < 2:
            recent_metrics = self.step_metrics
        
        time_span = recent_metrics[-1]['timestamp'] - recent_metrics[0]['timestamp']
        if time_span <= 0:
            return {'steps_per_sec': 0.0, 'tokens_per_sec': 0.0}
        
        steps_per_sec = len(recent_metrics) / time_span
        
        # Estimate tokens per second (assuming batch_size * seq_len per step)
        # This is approximate - actual implementation would need batch info
        tokens_per_sec = steps_per_sec * 32 * 512  # Default estimates
        
        return {
            'steps_per_sec': steps_per_sec,
            'tokens_per_sec': tokens_per_sec
        }
    
    def calculate_perplexity(self, loss: float) -> float:
        """Calculate perplexity from loss"""
        return math.exp(min(loss, 100))  # Cap to prevent overflow
    
    def get_progress_summary(self) -> Dict:
        """Get overall training progress summary"""
        total_time = time.time() - self.start_time
        
        summary = {
            'total_steps': len(self.step_metrics),
            'total_epochs': len(self.epoch_metrics),
            'total_stages': len(self.stage_metrics),
            'total_time': total_time,
            'best_train_loss': self.best_train_loss,
            'best_val_loss': self.best_val_loss,
            'best_val_perplexity': self.best_val_perplexity,
            'current_smoothed_loss': self.get_smoothed_loss(),
            'current_lr': self.get_smoothed_lr(),
        }
        
        # Add speed metrics
        summary.update(self.get_training_speed())
        
        return summary
    
    def get_stage_summary(self, stage_idx: int) -> Dict:
        """Get summary for specific stage"""
        stage_steps = [m for m in self.step_metrics if m.get('stage') == stage_idx]
        stage_epochs = [m for m in self.epoch_metrics if m['stage'] == stage_idx]
        
        if not stage_steps and not stage_epochs:
            return {}
        
        summary = {
            'stage_idx': stage_idx,
            'steps': len(stage_steps),
            'epochs': len(stage_epochs),
        }
        
        if stage_steps:
            losses = [s['loss'] for s in stage_steps]
            summary.update({
                'min_loss': min(losses),
                'max_loss': max(losses),
                'avg_loss': np.mean(losses),
                'final_loss': losses[-1],
            })
        
        if stage_epochs:
            val_losses = [e['val_loss'] for e in stage_epochs if e['val_loss'] is not None]
            if val_losses:
                summary.update({
                    'min_val_loss': min(val_losses),
                    'avg_val_loss': np.mean(val_losses),
                    'final_val_loss': val_losses[-1],
                })
        
        return summary
    
    def detect_overfitting(self, patience: int = 10, min_delta: float = 1e-4) -> bool:
        """Detect if model is overfitting"""
        if len(self.epoch_metrics) < patience:
            return False
        
        recent_epochs = self.epoch_metrics[-patience:]
        val_losses = [e['val_loss'] for e in recent_epochs if e['val_loss'] is not None]
        
        if len(val_losses) < patience:
            return False
        
        # Check if validation loss has not improved for 'patience' epochs
        best_loss = min(val_losses)
        recent_losses = val_losses[-patience//2:]  # Look at recent half
        
        # All recent losses should be worse than best by at least min_delta
        return all(loss > best_loss + min_delta for loss in recent_losses)
    
    def suggest_early_stopping(self, patience: int = 15, min_delta: float = 1e-4) -> bool:
        """Suggest early stopping based on validation metrics"""
        if len(self.epoch_metrics) < patience:
            return False
        
        recent_epochs = self.epoch_metrics[-patience:]
        val_losses = [e['val_loss'] for e in recent_epochs if e['val_loss'] is not None]
        
        if len(val_losses) < patience:
            return False
        
        # Check for improvement in recent epochs
        best_recent = min(val_losses)
        best_overall = self.best_val_loss
        
        # No improvement over best overall performance
        improvement = best_overall - best_recent
        return improvement < min_delta
    
    def export_metrics(self) -> Dict:
        """Export all metrics for saving/analysis"""
        return {
            'step_metrics': self.step_metrics,
            'epoch_metrics': self.epoch_metrics,
            'stage_metrics': self.stage_metrics,
            'summary': self.get_progress_summary(),
            'best_metrics': {
                'train_loss': self.best_train_loss,
                'val_loss': self.best_val_loss,
                'val_perplexity': self.best_val_perplexity,
            }
        }
    
    def load_metrics(self, metrics_data: Dict):
        """Load metrics from saved data"""
        self.step_metrics = metrics_data.get('step_metrics', [])
        self.epoch_metrics = metrics_data.get('epoch_metrics', [])
        self.stage_metrics = metrics_data.get('stage_metrics', [])
        
        best_metrics = metrics_data.get('best_metrics', {})
        self.best_train_loss = best_metrics.get('train_loss', float('inf'))
        self.best_val_loss = best_metrics.get('val_loss', float('inf'))
        self.best_val_perplexity = best_metrics.get('val_perplexity', float('inf'))


class LossTracker:
    """Specialized loss tracking with statistical analysis"""
    
    def __init__(self, name: str = "loss"):
        self.name = name
        self.values = []
        self.timestamps = []
        
    def update(self, value: float, timestamp: Optional[float] = None):
        """Add new loss value"""
        if timestamp is None:
            timestamp = time.time()
        
        self.values.append(value)
        self.timestamps.append(timestamp)
    
    def get_stats(self, recent_n: Optional[int] = None) -> Dict[str, float]:
        """Get statistical summary"""
        if not self.values:
            return {}
        
        values = self.values[-recent_n:] if recent_n else self.values
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'current': values[-1] if values else 0.0,
            'trend': self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float], window: int = 10) -> str:
        """Calculate trend direction"""
        if len(values) < window:
            return "insufficient_data"
        
        recent = values[-window:]
        older = values[-2*window:-window] if len(values) >= 2*window else values[:-window]
        
        if not older:
            return "insufficient_data"
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg < older_avg * 0.95:
            return "improving"
        elif recent_avg > older_avg * 1.05:
            return "degrading"
        else:
            return "stable"