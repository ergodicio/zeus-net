#!/bin/bash
#SBATCH -A m4490
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node 4
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND='cores';
export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/';
export MLFLOW_TRACKING_URI='/pscratch/sd/a/archis/mlflow';

# copy job stuff over
source /pscratch/sd/a/archis/venvs/zeus-gpu/bin/activate
module load cudnn/8.9.3_cuda12.lua

cd /global/u2/a/archis/zeus/