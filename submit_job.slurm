#!/bin/bash
#SBATCH --job-name=mcmult_mosi
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

# Load required modules (adjust based on your HPC system)
module purge
module load python/3.9
module load cuda/11.8
module load cudnn/8.6
module load nccl/2.16

# Activate virtual environment (if using)
# source /path/to/venv/bin/activate

# Or use conda environment
# conda activate mcmult_env

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0

# Print environment info
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"

# Create output directories
mkdir -p logs
mkdir -p checkpoints

# Run training with PyTorch distributed launcher
# For single node multi-GPU training
srun python -u train_hpc.py --config config.yaml

# Alternative: Use torchrun for PyTorch >= 1.10
# torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE \
#          --nnodes=$SLURM_NNODES \
#          --node_rank=$SLURM_NODEID \
#          --master_addr=$MASTER_ADDR \
#          --master_port=$MASTER_PORT \
#          train_hpc.py --config config.yaml

# Multi-node example (uncomment if using multiple nodes)
# srun python -u train_hpc.py --config config.yaml

echo "End Time: $(date)"
echo "Job completed successfully!"
