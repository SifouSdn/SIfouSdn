"""
Simple visualization script for monitoring MCMulT training progress
Reads metrics from logs/metrics.json and displays training curves
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(log_dir='./logs'):
    """Load metrics from JSON file"""
    metrics_file = os.path.join(log_dir, 'metrics.json')
    
    if not os.path.exists(metrics_file):
        print(f"Error: {metrics_file} not found. Run training first.")
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def plot_metrics(metrics, save_path='./logs/training_curves.png'):
    """Plot training metrics"""
    # Separate train and validation metrics
    train_metrics = [m for m in metrics if m['split'] == 'train']
    valid_metrics = [m for m in metrics if m['split'] == 'valid']
    
    if not train_metrics:
        print("No training metrics found.")
        return
    
    # Extract epochs
    train_epochs = [m['epoch'] for m in train_metrics]
    valid_epochs = [m['epoch'] for m in valid_metrics]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('MCMulT Training Progress', fontsize=16)
    
    # Loss
    axes[0, 0].plot(train_epochs, [m['loss'] for m in train_metrics], 'b-', label='Train')
    if valid_metrics:
        axes[0, 0].plot(valid_epochs, [m['loss'] for m in valid_metrics], 'r-', label='Valid')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # CCC (primary metric)
    axes[0, 1].plot(train_epochs, [m['ccc'] for m in train_metrics], 'b-', label='Train')
    if valid_metrics:
        axes[0, 1].plot(valid_epochs, [m['ccc'] for m in valid_metrics], 'r-', label='Valid')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('CCC')
    axes[0, 1].set_title('Concordance Correlation Coefficient')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].axhline(y=0.68, color='g', linestyle='--', label='Target Min')
    axes[0, 1].axhline(y=0.72, color='g', linestyle='--', label='Target Max')
    
    # MAE
    axes[0, 2].plot(train_epochs, [m['mae'] for m in train_metrics], 'b-', label='Train')
    if valid_metrics:
        axes[0, 2].plot(valid_epochs, [m['mae'] for m in valid_metrics], 'r-', label='Valid')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('MAE')
    axes[0, 2].set_title('Mean Absolute Error')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Correlation
    axes[1, 0].plot(train_epochs, [m['corr'] for m in train_metrics], 'b-', label='Train')
    if valid_metrics:
        axes[1, 0].plot(valid_epochs, [m['corr'] for m in valid_metrics], 'r-', label='Valid')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].set_title('Pearson Correlation')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 7-class Accuracy
    axes[1, 1].plot(train_epochs, [m['acc7'] for m in train_metrics], 'b-', label='Train')
    if valid_metrics:
        axes[1, 1].plot(valid_epochs, [m['acc7'] for m in valid_metrics], 'r-', label='Valid')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('7-Class Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Binary Accuracy
    axes[1, 2].plot(train_epochs, [m['acc2'] for m in train_metrics], 'b-', label='Train')
    if valid_metrics:
        axes[1, 2].plot(valid_epochs, [m['acc2'] for m in valid_metrics], 'r-', label='Valid')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Binary Accuracy')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.show()


def print_summary(metrics):
    """Print summary of training"""
    train_metrics = [m for m in metrics if m['split'] == 'train']
    valid_metrics = [m for m in metrics if m['split'] == 'valid']
    test_metrics = [m for m in metrics if m['split'] == 'test']
    
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    
    if train_metrics:
        print(f"\nTotal epochs: {len(train_metrics)}")
        print(f"Final train loss: {train_metrics[-1]['loss']:.4f}")
        print(f"Final train CCC: {train_metrics[-1]['ccc']:.4f}")
    
    if valid_metrics:
        best_epoch = max(range(len(valid_metrics)), key=lambda i: valid_metrics[i]['ccc'])
        print(f"\nBest validation epoch: {valid_metrics[best_epoch]['epoch']}")
        print(f"Best validation CCC: {valid_metrics[best_epoch]['ccc']:.4f}")
        print(f"Best validation MAE: {valid_metrics[best_epoch]['mae']:.4f}")
        print(f"Best validation Acc2: {valid_metrics[best_epoch]['acc2']:.4f}")
    
    if test_metrics:
        print(f"\nTest Results:")
        print(f"  CCC: {test_metrics[0]['ccc']:.4f}")
        print(f"  MAE: {test_metrics[0]['mae']:.4f}")
        print(f"  Corr: {test_metrics[0]['corr']:.4f}")
        print(f"  Acc7: {test_metrics[0]['acc7']:.4f}")
        print(f"  Acc2: {test_metrics[0]['acc2']:.4f}")
        print(f"  F1: {test_metrics[0]['f1']:.4f}")
        
        # Check if targets are met
        print("\nTarget Achievement:")
        target_met = 0.68 <= test_metrics[0]['ccc'] <= 0.72
        print(f"  CCC in target range [0.68, 0.72]: {'✓' if target_met else '✗'}")
    
    print("="*60 + "\n")


def main():
    """Main function"""
    print("MCMulT Training Visualization")
    print("-" * 60)
    
    # Load metrics
    metrics = load_metrics()
    if metrics is None:
        return
    
    print(f"Loaded {len(metrics)} metric entries")
    
    # Print summary
    print_summary(metrics)
    
    # Plot metrics
    try:
        plot_metrics(metrics)
    except Exception as e:
        print(f"Warning: Could not create plots. Error: {e}")
        print("You may need to install matplotlib: pip install matplotlib")


if __name__ == '__main__':
    main()
