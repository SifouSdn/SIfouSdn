#!/bin/bash
# Quick start script for MCMulT training
# This script demonstrates different ways to run the training

echo "MCMulT Training Quick Start Guide"
echo "=================================="
echo ""

# 1. Single GPU Training
echo "1. Single GPU Training:"
echo "   python train_hpc.py --config config.yaml"
echo ""

# 2. Multi-GPU Training (Single Node)
echo "2. Multi-GPU Training (Single Node, 4 GPUs):"
echo "   torchrun --nproc_per_node=4 train_hpc.py --config config.yaml"
echo "   OR"
echo "   python -m torch.distributed.launch --nproc_per_node=4 train_hpc.py --config config.yaml"
echo ""

# 3. HPC Cluster (SLURM)
echo "3. HPC Cluster Training (SLURM):"
echo "   sbatch submit_job.slurm"
echo ""

# 4. CPU-only Training (for testing)
echo "4. CPU-only Training (for testing):"
echo "   python train_hpc.py --config config.yaml"
echo ""

# Check if data exists
echo "Checking data directory..."
if [ ! -d "data/mosi" ]; then
    echo "Warning: data/mosi directory not found."
    echo "Creating data directory structure..."
    mkdir -p data/mosi
    echo "Note: The data loader will create dummy data for testing."
    echo "For actual training, place CMU-MOSI dataset files in data/mosi/:"
    echo "  - mosi_train.pkl"
    echo "  - mosi_valid.pkl"
    echo "  - mosi_test.pkl"
fi
echo ""

# Check if output directories exist
echo "Creating output directories..."
mkdir -p logs checkpoints
echo "Output directories created: logs/ and checkpoints/"
echo ""

# Run a quick test
echo "Would you like to run a quick test? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Running quick test with 2 epochs..."
    echo ""
    
    # Create test config
    cat > test_quick_start.yaml << 'EOF'
# Quick test configuration
data_path: './data/mosi'
max_seq_len: 50
audio_dim: 74
visual_dim: 35
text_dim: 300
d_model: 64
nhead: 4
num_layers: 2
dropout: 0.1
batch_size: 8
num_epochs: 2
learning_rate: 0.0001
weight_decay: 0.01
grad_clip: 1.0
optimizer: 'adamw'
scheduler: 'cosine'
num_workers: 0
log_dir: './logs'
checkpoint_dir: './checkpoints'
log_interval: 10
save_interval: 1
resume_checkpoint: null
seed: 42
EOF
    
    python train_hpc.py --config test_quick_start.yaml
    
    echo ""
    echo "Test completed! Check logs/ and checkpoints/ directories."
    echo "To view training log: cat logs/training.log"
    echo "To view metrics: cat logs/metrics.json"
    
    # Clean up test config
    rm -f test_quick_start.yaml
else
    echo "Skipping test. Run training manually using one of the commands above."
fi

echo ""
echo "For more information, see MCMulT_README.md"
