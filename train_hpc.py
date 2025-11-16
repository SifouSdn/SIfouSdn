"""
Training script for MCMulT with HPC distributed training support
Supports PyTorch DDP (DistributedDataParallel) for multi-GPU training
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from mcmult_model import create_model
from data_loader_hpc import create_dataloaders
from utils_hpc import (
    set_seed, setup_distributed, cleanup_distributed,
    save_checkpoint, load_checkpoint, calculate_metrics,
    Logger, AverageMeter, Timer, count_parameters
)


def train_epoch(model, train_loader, criterion, optimizer, epoch, config, logger, rank, device):
    """Train for one epoch"""
    model.train()
    loss_meter = AverageMeter()
    
    predictions = []
    labels = []
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        audio = batch['audio'].to(device, non_blocking=True)
        visual = batch['visual'].to(device, non_blocking=True)
        text = batch['text'].to(device, non_blocking=True)
        label = batch['label'].to(device, non_blocking=True)
        
        # Forward pass
        output = model(audio, visual, text)
        loss = criterion(output, label)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), audio.size(0))
        
        # Store predictions and labels
        predictions.extend(output.detach().cpu().numpy().flatten())
        labels.extend(label.detach().cpu().numpy().flatten())
        
        # Log progress
        if rank == 0 and batch_idx % config.get('log_interval', 10) == 0:
            logger.log(
                f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss_meter.avg:.4f}",
                print_msg=False
            )
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels)
    metrics['loss'] = loss_meter.avg
    
    return metrics


def validate(model, valid_loader, criterion, epoch, config, logger, rank, device):
    """Validate model"""
    model.eval()
    loss_meter = AverageMeter()
    
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in valid_loader:
            # Move data to device
            audio = batch['audio'].to(device, non_blocking=True)
            visual = batch['visual'].to(device, non_blocking=True)
            text = batch['text'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)
            
            # Forward pass
            output = model(audio, visual, text)
            loss = criterion(output, label)
            
            # Update metrics
            loss_meter.update(loss.item(), audio.size(0))
            
            # Store predictions and labels
            predictions.extend(output.cpu().numpy().flatten())
            labels.extend(label.cpu().numpy().flatten())
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels)
    metrics['loss'] = loss_meter.avg
    
    return metrics


def test(model, test_loader, criterion, config, logger, rank, device):
    """Test model"""
    model.eval()
    loss_meter = AverageMeter()
    
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            audio = batch['audio'].to(device, non_blocking=True)
            visual = batch['visual'].to(device, non_blocking=True)
            text = batch['text'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)
            
            # Forward pass
            output = model(audio, visual, text)
            loss = criterion(output, label)
            
            # Update metrics
            loss_meter.update(loss.item(), audio.size(0))
            
            # Store predictions and labels
            predictions.extend(output.cpu().numpy().flatten())
            labels.extend(label.cpu().numpy().flatten())
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels)
    metrics['loss'] = loss_meter.avg
    
    return metrics


def main_worker(rank, world_size, config):
    """Main training worker for each GPU"""
    
    # Setup distributed training
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # Set random seed for reproducibility
    set_seed(config['seed'] + rank)
    
    # Create logger
    logger = Logger(config['log_dir'], rank=rank)
    
    if rank == 0:
        logger.log(f"Starting training with {world_size} GPUs")
        logger.log_config(config)
    
    # Create dataloaders
    train_loader, valid_loader, test_loader = create_dataloaders(
        config, 
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size
    )
    
    # Create model
    model = create_model(config)
    
    # Move to device (GPU if available, otherwise CPU)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if rank == 0:
        logger.log(f"Model parameters: {count_parameters(model):,}")
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0)
        )
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    # Learning rate scheduler
    scheduler = None
    if config.get('scheduler') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs']
        )
    elif config.get('scheduler') == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 10),
            gamma=config.get('gamma', 0.5)
        )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_ccc = -float('inf')
    
    if config.get('resume_checkpoint'):
        if os.path.exists(config['resume_checkpoint']):
            if rank == 0:
                logger.log(f"Resuming from checkpoint: {config['resume_checkpoint']}")
            start_epoch, metrics = load_checkpoint(
                config['resume_checkpoint'],
                model.module if world_size > 1 else model,
                optimizer,
                scheduler
            )
            best_ccc = metrics.get('best_ccc', -float('inf'))
            start_epoch += 1
    
    # Training loop
    timer = Timer()
    
    for epoch in range(start_epoch, config['num_epochs']):
        timer.start()
        
        # Set epoch for distributed sampler
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, 
            epoch, config, logger, rank, device
        )
        
        if rank == 0:
            logger.log_metrics(epoch, 'train', train_metrics)
        
        # Validate
        valid_metrics = validate(
            model, valid_loader, criterion, 
            epoch, config, logger, rank, device
        )
        
        if rank == 0:
            logger.log_metrics(epoch, 'valid', valid_metrics)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Save checkpoint
        if rank == 0:
            # Save best model based on CCC
            if valid_metrics['ccc'] > best_ccc:
                best_ccc = valid_metrics['ccc']
                save_checkpoint(
                    model.module if world_size > 1 else model,
                    optimizer,
                    scheduler,
                    epoch,
                    {'best_ccc': best_ccc, **valid_metrics},
                    config,
                    os.path.join(config['checkpoint_dir'], 'best_model.pt')
                )
                logger.log(f"New best model saved with CCC: {best_ccc:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % config.get('save_interval', 10) == 0:
                save_checkpoint(
                    model.module if world_size > 1 else model,
                    optimizer,
                    scheduler,
                    epoch,
                    valid_metrics,
                    config,
                    os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pt')
                )
        
        epoch_time = timer.stop()
        if rank == 0:
            logger.log(f"Epoch {epoch} completed in {epoch_time:.2f}s")
    
    # Final testing
    if rank == 0:
        logger.log("Training completed. Running final test...")
        
        # Load best model
        best_model_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
        if os.path.exists(best_model_path):
            _, _ = load_checkpoint(
                best_model_path,
                model.module if world_size > 1 else model
            )
        
        # Test
        test_metrics = test(
            model, test_loader, criterion, 
            config, logger, rank, device
        )
        
        logger.log_metrics(-1, 'test', test_metrics)
        logger.log(f"Final test results - CCC: {test_metrics['ccc']:.4f}, MAE: {test_metrics['mae']:.4f}")
    
    # Cleanup
    if world_size > 1:
        cleanup_distributed()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train MCMulT on CMU-MOSI with HPC support')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Get world size
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
    else:
        world_size = torch.cuda.device_count()
        rank = args.local_rank
    
    if world_size == 0:
        print("No GPUs available. Running on CPU.")
        world_size = 1
        rank = 0
    
    # Run training
    if world_size > 1:
        main_worker(rank, world_size, config)
    else:
        main_worker(0, 1, config)


if __name__ == '__main__':
    main()
