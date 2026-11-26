#!/bin/bash -l
# Training benchmark on one GPU in the ucmf container; arguments go to benchmark_training.py.
#   sbatch --constraint=tesla_a100 benchmark_training.sbatch --mics 16
#   sbatch --constraint=tesla_p100 benchmark_training.sbatch --mics 16 --mode eager  # no Triton on sm_60
# UCMF (repository) and SIF (image built from ucmf.def) default to the scratch layout below.
#SBATCH --job-name=ucmf_bench
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out

UCMF=${UCMF:-/beegfs/scratch/$USER/ucmf_bench/ucmf}
SIF=${SIF:-/beegfs/scratch/$USER/ucmf_bench/ucmf.sif}
module load singularity/4.4.0
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
singularity exec --nv --bind "$UCMF:/opt/ucmf" "$SIF" python /opt/ucmf/examples/benchmark_training.py "$@"
