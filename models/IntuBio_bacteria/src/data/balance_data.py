"""
balance_data.py balances the unbalanced bacteria dataset by 
stochastically removing part of the dataset that has no bacteria present
in the images. 

Args:
    - image_path: Path to the raw images in the training data
    - mask_path: Path to the ground truth images in the training data
    - rem_prob: Probability of removing a given pair of image/ground 
                truth from the data set containing no bacteria
    - output_path: Path to the folder in which the filtered dataset should
                   be located

Returns:
    - New data set in specified folder "output_path"
"""

from patchify import patchify
import cv2
import os
import numpy as np
from PIL import Image

# NxN image patches with stepsize 256, means 50% overlap
N=512
step=256

# save probability 1-prob
prob = 0.85

# Flags - set True on the wanted flag to clean the data
TRAIN = True
VAL = False

if TRAIN:
    # Main image directory
    main_directory="/work3/s174182/train_data/Annotated_segmentation_patch/train/"

    # Save directory to
    save_directory="/work3/s174182/train_data/Annotated_segmentation_patch_balanced/train/"
    VAL = False

elif VAL:
    # Main image directory
    main_directory="/work3/s174182/train_data/Annotated_segmentation_patch/val/"

    # Save directory to
    save_directory="/work3/s174182/train_data/Annotated_segmentation_patch_balanced/val/"
    TRAIN = False
    
# Indices to save
idx_save = []

# Get in the main directory and go through the samples
folders=os.listdir(main_directory)
for f in folders:
    # Go through wells A-C
    sub_folders=os.listdir(os.path.join(main_directory,f))
    for sf in sub_folders:
        # Go through masks and check if for empty images
        for masks in os.listdir(os.path.join(main_directory, f, sf, "mask/")):
            # Read image and check if they are black
            mask = cv2.imread(os.path.join(main_directory, f, sf, "mask/", masks), 0)
            if mask.sum() < 1:
                # If the image is black, draw uniform random number, 
                # and add mask + raw image to save folder if that number is above prob
                rng_num = np.random.uniform(low=0.0, high=1.0)
                if rng_num < prob:
                    continue
                
                else:
                    # Get image index
                    idx = masks.split('_')[1][:-4]

                    # Save mask
                    os.makedirs(os.path.join(save_directory, f, sf, "mask/"), exist_ok=True)
                    cv2.imwrite(os.path.join(save_directory, f, sf, "mask/", masks), mask)
                    
                    # Save rwa image
                    os.makedirs(os.path.join(save_directory, f, sf, "img/"), exist_ok=True)
                    saveimg = cv2.imread(os.path.join(main_directory, f, sf, "img/", "NormIP_"+str(idx)+".png"), 0)
                    cv2.imwrite(os.path.join(save_directory, f, sf, "img/", "NormIP_"+str(idx)+".png"), saveimg)
                
            else:
                # Get index
                idx = masks.split('_')[1][:-4]
                # Save mask
                os.makedirs(os.path.join(save_directory, f, sf, "mask/"), exist_ok=True) # create save directory
                cv2.imwrite(os.path.join(save_directory, f, sf, "mask/", masks), mask) # save in save directory
                # Save raw
                os.makedirs(os.path.join(save_directory, f, sf, "img/"), exist_ok=True)
                saveimg = cv2.imread(os.path.join(main_directory, f, sf, "img/", "NormIP_"+str(idx)+".png"), 0)
                cv2.imwrite(os.path.join(save_directory, f, sf, "img/", "NormIP_"+str(idx)+".png"), saveimg)
                
