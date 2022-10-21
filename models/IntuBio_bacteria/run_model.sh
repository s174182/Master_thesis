#!/bin/bash
#BSUB -J endpoint
#BSUB -o endpoint_%J.out
#BSUB -e endpoint_%J.err
#BSUB -q gpua100
#BSUB -R "rusage[mem=32GB]"
## NUMBER OF CORES
#BSUB -n 1 #IF USING ALL THREADS POSSIBLE
#BSUB -W 23:30
#BSUB -N
#BSUB -gpu "num=1"
#BSUB -R "span[hosts=1]"

load module python3/10.3.2
load module cuda
source ~/endpoint_segmentation/bin/activate
# Get api key for wandb
export WANDB_API_KEY=$(cat wandbkey.txt)
wandb login #wandb login

# Start sweep from config.yaml
#wandb agent ${sweep_id}
wandb agent --count $NUM s174182/Master_thesis-models_IntuBio_bacteria_src_models/vhv2aqmf


python3 src/models/train_model.py