#!/bin/bash
#SBATCH --job-name=Taxipro
#SBATCH --output=logs/Taxipro_%j.out
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
module load miniforge3/25.11.0-1
module load cuda/12.8 # 服务器内置了11.8,12.1,12.4,12.8,12.9,13.0,请根据实际需求选择
source activate pytorch5090

## 需要执行的代码
export TQDM_DISABLE=1
python main.py -d Taxipro --model_name deepseek-7b --seed 39 --batch_size 16 --epoch 50 --patience 20