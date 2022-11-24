#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:43:09 2022

@author: frederikhartmann

Data set class

"""

import numpy as np
import os
from torch.utils.data import Dataset
import glob
from PIL import Image
import albumentations as A



class BacteriaDataset(Dataset):
    # Initialize
    def __init__(self, image_dir, mask_dir, transform=None, skipborders=False):
        # List subfolders with glob
        file_list = glob.glob(image_dir + "*")
        
        # Allocate for images and masks
        self.images = []
        self.masks = []
        
        #Go through each folder in subfolder
        for f in file_list:
            # Go through A1-C4
            for sf in os.listdir(f):   
                if ".DS_Store" in sf:
                    continue
                # append inorm to images and mask to masks
                img_folder=os.listdir(f + "/" + sf + "/img")
                mask_folder=os.listdir(f + "/" + sf + "/mask")
                for k in range(len(img_folder)):

                    # Skip masks with CFU on border
                    if checkborder(f + "/" + sf + "/mask/"+mask_folder[k]) and skipborders:
                        continue
                    
                    if np.sum(np.array(Image.open(f + "/" + sf +"/img/"+img_folder[k]))) < 1:
                        continue

                    self.images.append(f + "/" + sf +"/img/"+img_folder[k])
                    self.masks.append(f + "/" + sf + "/mask/"+mask_folder[k])
         
        self.images.sort() 
        self.masks.sort()      

        self.transform = transform
        
    # Define length of dataset
    def __len__(self):
        return len(self.images)


    # Define get item for data loader
    def __getitem__(self, index):
        # get image
        img_path = self.images[index]
        image = np.array(Image.open(img_path))            

        # Normalize image
        im_mean1, im_std1 = np.mean(image[:,:,0]), np.std(image[:,:,0])
        im_mean2, im_std2 = np.mean(image[:,:,1]), np.std(image[:,:,1])
        im_mean3, im_std3 = np.mean(image[:,:,2]), np.std(image[:,:,2])

        # Normalize image as to match training
        normalize_transform = A.Compose([A.augmentations.transforms.Normalize(mean=(im_mean1, im_mean2, im_mean3), std=(im_std1, im_std2, im_std3), max_pixel_value=1, always_apply=False, p=1.0),
        ])
        augmented = normalize_transform(image=image)
        image=augmented["image"]

        # get mask
        mask_path = self.masks[index]
        mask = np.array(Image.open(mask_path).convert("L"))

        # Preprocess mask - convert to 1
        mask[mask != 0] = 1.0

        # Do transforms if needed
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # cv2.imshow('img', image.numpy().squeeze())
        # cv2.imshow("mask",mask.numpy())
        # cv2.waitKey()
        return image, mask

def checkborder(mask_path):
    # Read image
    mask = np.array(Image.open(mask_path).convert("L"))
    # Preprocess mask - convert to 1
    mask[mask != 0] = 1.0
    skip = False

    # If there is a bacteria on the edge of the mask, we skip
    if np.sum(mask[:,:4] > 1) or np.sum(mask[:,-4:] > 1) or np.sum(mask[:4,:] > 1) or np.sum(mask[-4:,:] > 1):
        skip = True

    return skip