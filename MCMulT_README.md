# MCMulT: Multi-scale Cooperative Multimodal Transformer

Implementation of MCMulT for CMU-MOSI sentiment analysis with HPC distributed training support.

## Overview

MCMulT is a multimodal transformer architecture designed for sentiment analysis on the CMU-MOSI (Multimodal Opinion Sentiment Intensity) dataset. The model processes three modalities:
- **Audio**: COVAREP features (74-dim)
- **Visual**: Facet features (35-dim)  
- **Text**: GloVe embeddings (300-dim)

### Key Features

- **Multi-scale temporal convolution**: Captures patterns at different time scales
- **Cross-modal attention**: Enables interaction between all modality pairs
- **Cooperative fusion**: Dynamically weights and combines multimodal features
- **HPC support**: Distributed training with PyTorch DDP for multi-GPU acceleration
- **Reproducible results**: Deterministic seeding for consistent training outcomes

### Performance Targets

- **CCC (Concordance Correlation Coefficient)**: 0.68-0.72
- **Multi-GPU speedup**: 10-50x with distributed training
- **Reproducibility**: Deterministic operations for consistent results

## Project Structure

```
.
├── mcmult_model.py         # MCMulT model architecture
├── train_hpc.py            # Training script with DDP support
├── data_loader_hpc.py      # Efficient data loading for CMU-MOSI
├── utils_hpc.py            # Utilities (logging, checkpointing, metrics)
├── config.yaml             # Hyperparameter configuration
├── submit_job.slurm        # SLURM job submission script
└── requirements.txt        # Python dependencies
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU support)
- NCCL 2.0+ (for multi-GPU training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SifouSdn/SIfouSdn.git
cd SIfouSdn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the CMU-MOSI dataset:
```bash
# Create data directory
mkdir -p data/mosi

# Download and extract CMU-MOSI dataset
# Place the following files in data/mosi/:
#   - mosi_train.pkl
#   - mosi_valid.pkl
#   - mosi_test.pkl
```

Note: If dataset files are not available, the data loader will create dummy data for testing.

## Usage

### Single GPU Training

```bash
python train_hpc.py --config config.yaml
```

### Multi-GPU Training (Single Node)

Using `torchrun` (PyTorch >= 1.10):
```bash
torchrun --nproc_per_node=4 train_hpc.py --config config.yaml
```

Or using environment variables:
```bash
WORLD_SIZE=4 python train_hpc.py --config config.yaml
```

### HPC Cluster Training (SLURM)

Submit job to SLURM:
```bash
sbatch submit_job.slurm
```

Monitor job:
```bash
squeue -u $USER
tail -f logs/slurm_<job_id>.out
```

### Configuration

Edit `config.yaml` to customize hyperparameters:

```yaml
# Model architecture
d_model: 128          # Hidden dimension
nhead: 8              # Number of attention heads
num_layers: 4         # Number of transformer layers
dropout: 0.1          # Dropout rate

# Training settings
batch_size: 32        # Batch size per GPU
num_epochs: 100       # Total epochs
learning_rate: 0.0001 # Learning rate
optimizer: 'adamw'    # Optimizer (adam, adamw)
scheduler: 'cosine'   # LR scheduler (cosine, step)

# Reproducibility
seed: 42              # Random seed
```

## Model Architecture

### Multi-scale Temporal Convolution

Captures temporal patterns at different scales using parallel 1D convolutions with kernel sizes [3, 5, 7]:

```python
MultiScaleTemporalConv(in_channels, out_channels, kernel_sizes=[3, 5, 7])
```

### Cross-modal Attention

Each modality attends to the other two modalities using multi-head attention:

```
Audio → Text, Audio → Visual
Visual → Text, Visual → Audio  
Text → Audio, Text → Visual
```

### Cooperative Fusion

Dynamically combines modality features using learned attention weights and gating:

```python
fused = weighted_sum(modalities) * gate(concatenated_features)
```

## Evaluation Metrics

The model is evaluated using multiple metrics:

- **CCC**: Concordance Correlation Coefficient (primary metric)
- **MAE**: Mean Absolute Error
- **Corr**: Pearson Correlation
- **Acc7**: 7-class classification accuracy
- **Acc2**: Binary classification accuracy (positive/negative)
- **F1**: F1 score for binary classification

## Results

Expected performance on CMU-MOSI test set:

| Metric | Target | Description |
|--------|--------|-------------|
| CCC    | 0.68-0.72 | Concordance correlation coefficient |
| MAE    | ~0.8-0.9  | Mean absolute error |
| Acc7   | ~32-35%   | 7-class accuracy |
| Acc2   | ~78-82%   | Binary accuracy |

## Reproducibility

The implementation ensures reproducible results through:

1. **Deterministic seeding**: Random seeds set for Python, NumPy, and PyTorch
2. **Deterministic operations**: `torch.backends.cudnn.deterministic = True`
3. **Consistent initialization**: Fixed random state for weight initialization
4. **Synchronized randomness**: Different seeds for different GPU ranks

## Distributed Training Performance

Expected speedup with multi-GPU training:

| GPUs | Speedup | Efficiency |
|------|---------|------------|
| 1    | 1x      | 100%       |
| 2    | 1.8-1.9x| 90-95%     |
| 4    | 3.5-3.8x| 87-95%     |
| 8    | 6.5-7.5x| 81-94%     |

Factors affecting speedup:
- Batch size per GPU
- Communication overhead (NCCL)
- Data loading bottlenecks
- GPU interconnect (NVLink vs PCIe)

## Checkpoints and Logging

### Checkpoints

Saved in `checkpoints/` directory:
- `best_model.pt`: Best model based on validation CCC
- `checkpoint_epoch_N.pt`: Periodic checkpoints every N epochs

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Training metrics
- Configuration

### Logs

Saved in `logs/` directory:
- `training.log`: Training progress and metrics
- `metrics.json`: Metrics history in JSON format
- `config.json`: Training configuration
- `slurm_<job_id>.out`: SLURM job output

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size in `config.yaml`:
```yaml
batch_size: 16  # or smaller
```

### Slow Data Loading

Increase number of workers:
```yaml
num_workers: 8  # adjust based on CPU cores
```

### Poor Convergence

Try adjusting learning rate and scheduler:
```yaml
learning_rate: 0.0005
scheduler: 'step'
step_size: 20
gamma: 0.5
```

### Multi-GPU Training Issues

Check NCCL environment:
```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # or ib0 for InfiniBand
```

## Citation

If you use this code, please cite:

```bibtex
@misc{mcmult2024,
  author = {Seif-Allah Saidoun},
  title = {MCMulT: Multi-scale Cooperative Multimodal Transformer},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/SifouSdn/SIfouSdn}
}
```

## References

1. CMU-MOSI Dataset: [MultimodalDataset](http://multicomp.cs.cmu.edu/)
2. PyTorch Distributed: [DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
3. Transformer Architecture: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## License

This project is licensed under the MIT License.

## Contact

For questions or collaborations:
- **Email**: seif-allah.saidoun@ensia.edu.dz
- **LinkedIn**: [Seif-Allah Saidoun](https://www.linkedin.com/in/seif-allah-saidoun-116646246/)
- **GitHub**: [SifouSdn](https://github.com/SifouSdn)
