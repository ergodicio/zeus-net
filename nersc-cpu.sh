#!/bin/bash
#SBATCH --qos=regular
#SBATCH -A m4490
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=cpu

export BASE_TEMPDIR='/pscratch/sd/a/archis/tmp/'
export MLFLOW_TRACKING_URI='/pscratch/sd/a/archis/mlflow'
export MLFLOW_EXPORT=True
export MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=False

source /pscratch/sd/a/archis/venvs/zeus-gpu/bin/activate
cd /global/u2/a/archis/zeus/
srun python ta3_model.py