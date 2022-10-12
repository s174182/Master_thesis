#!/bin/bash
#BSUB -J endpoint
#BSUB -o endpoint_%J.out
#BSUB -e endpoint_%J.err
#BSUB -q gpua100
#BSUB -R "rusage[mem=8GB]"
## NUMBER OF CORES
#BSUB -n 8 #IF USING ALL THREADS POSSIBLE
#BSUB -W 22:00
#BSUB -N
#BSUB -gpu "num=1"
#BSUB -R "span[hosts=1]"
load module python3/10.3.2
load module cuda
source ~/endpoint_segmentation/bin/activate
# Get api key for wandb
export WANDB_API_KEY=effe53cafb09230e54bd97ef4cb5c393eac33c74
wandb login #wandb login

python3 ./src/models/train_model.py