"""
Utility functions for HPC training
Includes logging, checkpointing, metrics calculation, and reproducibility
"""

import os
import json
import time
import random
import numpy as np
import torch
import torch.distributed as dist
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, accuracy_score


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, filename):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config
    }
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """Load training checkpoint"""
    checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']


def calculate_metrics(predictions, labels):
    """
    Calculate evaluation metrics for sentiment analysis
    
    Metrics:
    - MAE: Mean Absolute Error
    - Corr: Pearson Correlation
    - Acc7: 7-class accuracy
    - Acc2: Binary accuracy
    - F1: F1 score for binary classification
    """
    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()
    
    # MAE
    mae = mean_absolute_error(labels, predictions)
    
    # Pearson Correlation (CCC approximation)
    corr, _ = pearsonr(predictions, labels) if len(predictions) > 1 else (0, 1)
    
    # Convert to 7 classes [-3, -2, -1, 0, 1, 2, 3]
    pred_7 = np.clip(np.round(predictions), -3, 3).astype(int)
    label_7 = np.clip(np.round(labels), -3, 3).astype(int)
    acc7 = accuracy_score(label_7, pred_7)
    
    # Convert to binary [negative, positive]
    pred_binary = (predictions >= 0).astype(int)
    label_binary = (labels >= 0).astype(int)
    acc2 = accuracy_score(label_binary, pred_binary)
    
    # Calculate F1 score for binary classification
    from sklearn.metrics import f1_score
    f1 = f1_score(label_binary, pred_binary, average='weighted', zero_division=0)
    
    # CCC (Concordance Correlation Coefficient)
    mean_pred = np.mean(predictions)
    mean_label = np.mean(labels)
    var_pred = np.var(predictions)
    var_label = np.var(labels)
    cov = np.cov(predictions, labels)[0, 1]
    ccc = (2 * cov) / (var_pred + var_label + (mean_pred - mean_label) ** 2)
    
    metrics = {
        'mae': mae,
        'corr': corr,
        'ccc': ccc,
        'acc7': acc7,
        'acc2': acc2,
        'f1': f1
    }
    
    return metrics


class Logger:
    """Logger for training progress"""
    
    def __init__(self, log_dir, rank=0):
        self.log_dir = log_dir
        self.rank = rank
        self.is_main = (rank == 0)
        
        if self.is_main:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, 'training.log')
            self.metrics_file = os.path.join(log_dir, 'metrics.json')
            self.metrics_history = []
    
    def log(self, message, print_msg=True):
        """Log message to file and console"""
        if not self.is_main:
            return
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        if print_msg:
            print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def log_metrics(self, epoch, split, metrics, print_msg=True):
        """Log metrics for an epoch"""
        if not self.is_main:
            return
        
        # Convert numpy types to native Python types for JSON serialization
        metrics_clean = {}
        for k, v in metrics.items():
            if hasattr(v, 'item'):
                metrics_clean[k] = v.item()
            elif isinstance(v, np.ndarray):
                metrics_clean[k] = v.tolist()
            elif isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                metrics_clean[k] = float(v) if isinstance(v, (np.float32, np.float64)) else int(v)
            else:
                metrics_clean[k] = v
        
        metrics_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metrics_clean.items()])
        self.log(f"Epoch {epoch} - {split}: {metrics_str}", print_msg)
        
        # Save to history
        metrics_entry = {
            'epoch': epoch,
            'split': split,
            'timestamp': datetime.now().isoformat(),
            **metrics_clean
        }
        self.metrics_history.append(metrics_entry)
        
        # Save metrics to JSON
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def log_config(self, config):
        """Log configuration"""
        if not self.is_main:
            return
        
        config_file = os.path.join(self.log_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.log("Configuration:")
        for key, value in config.items():
            self.log(f"  {key}: {value}")


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Simple timer for measuring execution time"""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        if self.start_time:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def get_gpu_memory():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    """Format time in seconds to human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == '__main__':
    # Test utilities
    set_seed(42)
    
    # Test metrics calculation
    predictions = np.random.randn(100) * 2
    labels = np.random.randn(100) * 2
    metrics = calculate_metrics(predictions, labels)
    print("Test metrics:", metrics)
    
    # Test logger
    logger = Logger('./logs', rank=0)
    logger.log("Test log message")
    logger.log_metrics(1, 'train', {'loss': 0.5, 'acc': 0.8})
