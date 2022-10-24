#!/bin/bash
#BSUB -J endpoint
#BSUB -o endpoint_%J.out
#BSUB -e endpoint_%J.err
#BSUB -q gpua100
#BSUB -R "rusage[mem=3GB]"
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

export WANDB_API_KEY=$(cat wandbkey.txt)

wandb login #wandb login
NUM=10
wandb agent --count $NUM intubio/Master_thesis-models_IntuBio_bacteria_rgb_2_src_models/doc445z2 # sweep agent

python3 ./src/models/train_model_sweep.py