#!/bin/bash
#SBATCH --qos=training
#SBATCH -D .
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.err
#SBATCH --cpus-per-task=80			
#SBATCH --gres gpu:4
#SBATCH --time=24:00:00
module purge; module load tensorflow
python model4.py

