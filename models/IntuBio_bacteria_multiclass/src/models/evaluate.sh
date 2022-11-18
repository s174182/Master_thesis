#!/bin/bash
#BSUB -J endpoint
#BSUB -o endpoint_%J.out
#BSUB -e endpoint_%J.err
#BSUB -q gpua100
#BSUB -R "rusage[mem=24GB]"
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

python3 evaluate.py