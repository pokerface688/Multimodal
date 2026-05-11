#!/bin/bash
#SBATCH --job-name=data_embed
#SBATCH --output=logs/data_embed_%j.out
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
module load miniforge3/25.11.0-1
module load cuda/12.8 # 服务器内置了11.8,12.1,12.4,12.8,12.9,13.0,请根据实际需求选择
source activate pytorch5090

## 需要执行的代码
export TQDM_DISABLE=1
python merge_embedding.py
