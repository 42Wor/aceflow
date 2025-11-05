import numpy as np
from typing import Dict, List, Callable
import torch

class Metric:
    """Base metric class"""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        self.values = []
    
    def update(self, value):
        self.values.append(value)
    
    def compute(self):
        return np.mean(self.values) if self.values else 0.0

class MetricTracker:
    """Tracks multiple metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def add_metric(self, name: str, metric: Metric):
        self.metrics[name] = metric
    
    def update(self, name: str, value):
        if name not in self.metrics:
            self.metrics[name] = Metric(name)
        self.metrics[name].update(value)
    
    def compute_all(self) -> Dict[str, float]:
        return {name: metric.compute() for name, metric in self.metrics.items()}
    
    def reset_all(self):
        for metric in self.metrics.values():
            metric.reset()

class AccuracyMetric(Metric):
    """Accuracy metric for classification"""
    
    def __init__(self, ignore_index: int = -100):
        super().__init__('accuracy')
        self.ignore_index = ignore_index
        self.correct = 0
        self.total = 0
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(self, predictions, targets, mask=None):
        if mask is None:
            mask = targets != self.ignore_index
        
        self.correct += ((predictions == targets) & mask).sum().item()
        self.total += mask.sum().item()
    
    def compute(self):
        return self.correct / self.total if self.total > 0 else 0.0

class LossMetric(Metric):
    """Loss metric"""
    
    def __init__(self):
        super().__init__('loss')
        self.total_loss = 0
        self.total_batches = 0
    
    def reset(self):
        self.total_loss = 0
        self.total_batches = 0
    
    def update(self, loss, batch_size=1):
        self.total_loss += loss * batch_size
        self.total_batches += batch_size
    
    def compute(self):
        return self.total_loss / self.total_batches if self.total_batches > 0 else 0.0