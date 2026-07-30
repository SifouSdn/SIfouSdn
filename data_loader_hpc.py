"""
Efficient data loader for CMU-MOSI dataset with HPC optimizations
Supports distributed training with PyTorch DDP
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import h5py


class CMUMOSIDataset(Dataset):
    """
    CMU-MOSI (Multimodal Opinion Sentiment Intensity) Dataset
    
    Features:
    - Audio: COVAREP features (74-dim)
    - Visual: Facet features (35-dim)
    - Text: GloVe embeddings (300-dim)
    - Labels: Sentiment intensity [-3, 3]
    """
    
    def __init__(self, data_path, split='train', max_seq_len=50, use_cache=True):
        """
        Args:
            data_path: Path to MOSI data directory
            split: 'train', 'valid', or 'test'
            max_seq_len: Maximum sequence length for padding/truncating
            use_cache: Whether to use cached preprocessed data
        """
        self.data_path = data_path
        self.split = split
        self.max_seq_len = max_seq_len
        self.use_cache = use_cache
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self):
        """Load and preprocess CMU-MOSI data"""
        cache_file = os.path.join(self.data_path, f'mosi_{self.split}_cache.pkl')
        
        if self.use_cache and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Load raw data (assumes standard MOSI format)
        data_file = os.path.join(self.data_path, f'mosi_{self.split}.pkl')
        
        if not os.path.exists(data_file):
            # Create dummy data for testing
            print(f"Warning: {data_file} not found. Creating dummy data for testing.")
            return self._create_dummy_data()
        
        with open(data_file, 'rb') as f:
            raw_data = pickle.load(f)
        
        # Preprocess data
        processed_data = []
        for item in raw_data:
            processed_item = {
                'audio': self._pad_or_truncate(item['audio'], self.max_seq_len),
                'visual': self._pad_or_truncate(item['visual'], self.max_seq_len),
                'text': self._pad_or_truncate(item['text'], self.max_seq_len),
                'label': item['label']
            }
            processed_data.append(processed_item)
        
        # Cache processed data
        if self.use_cache:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(processed_data, f)
        
        return processed_data
    
    def _create_dummy_data(self):
        """Create dummy data for testing when real data is not available"""
        np.random.seed(42)
        n_samples = {'train': 1000, 'valid': 200, 'test': 200}[self.split]
        
        data = []
        for i in range(n_samples):
            # Random features
            audio = np.random.randn(self.max_seq_len, 74).astype(np.float32)
            visual = np.random.randn(self.max_seq_len, 35).astype(np.float32)
            text = np.random.randn(self.max_seq_len, 300).astype(np.float32)
            
            # Random sentiment label between -3 and 3
            label = np.random.uniform(-3, 3)
            
            data.append({
                'audio': audio,
                'visual': visual,
                'text': text,
                'label': label
            })
        
        return data
    
    def _pad_or_truncate(self, sequence, max_len):
        """Pad or truncate sequence to max_len"""
        seq_len, feat_dim = sequence.shape
        
        if seq_len > max_len:
            # Truncate
            return sequence[:max_len]
        elif seq_len < max_len:
            # Pad with zeros
            padding = np.zeros((max_len - seq_len, feat_dim), dtype=sequence.dtype)
            return np.vstack([sequence, padding])
        else:
            return sequence
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert to tensors
        audio = torch.FloatTensor(item['audio'])
        visual = torch.FloatTensor(item['visual'])
        text = torch.FloatTensor(item['text'])
        label = torch.FloatTensor([item['label']])
        
        return {
            'audio': audio,
            'visual': visual,
            'text': text,
            'label': label
        }


def create_dataloaders(config, distributed=False, rank=0, world_size=1):
    """
    Create train, validation, and test dataloaders
    
    Args:
        config: Configuration dictionary
        distributed: Whether to use distributed sampling
        rank: Process rank for distributed training
        world_size: Total number of processes
    
    Returns:
        train_loader, valid_loader, test_loader
    """
    data_path = config.get('data_path', './data/mosi')
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    max_seq_len = config.get('max_seq_len', 50)
    
    # Create datasets
    train_dataset = CMUMOSIDataset(data_path, split='train', max_seq_len=max_seq_len)
    valid_dataset = CMUMOSIDataset(data_path, split='valid', max_seq_len=max_seq_len)
    test_dataset = CMUMOSIDataset(data_path, split='test', max_seq_len=max_seq_len)
    
    # Create samplers for distributed training
    train_sampler = None
    valid_sampler = None
    test_sampler = None
    
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.get('seed', 42)
        )
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"Dataset sizes - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    # Test dataloader
    config = {
        'data_path': './data/mosi',
        'batch_size': 4,
        'num_workers': 0,
        'max_seq_len': 50
    }
    
    train_loader, valid_loader, test_loader = create_dataloaders(config)
    
    # Test one batch
    for batch in train_loader:
        print("Batch shapes:")
        print(f"  Audio: {batch['audio'].shape}")
        print(f"  Visual: {batch['visual'].shape}")
        print(f"  Text: {batch['text'].shape}")
        print(f"  Label: {batch['label'].shape}")
        break
