#!/bin/bash
#BSUB -J mitochondria_test
#BSUB -o mitochondria_%J.out
#BSUB -e mitochindria_%J.err
#BSUB -q gpuv100
#BSUB -R "rusage[mem=12GB]"
#BSUB -n 1 #IF USING ALL THREADS POSSIBLE
#BSUB -W 00:30
#BSUB -N
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
load module python3/10.3.2
load module cuda
source ~/endpoint_segmentation/bin/activate


python3 ./src/models/train_model.py