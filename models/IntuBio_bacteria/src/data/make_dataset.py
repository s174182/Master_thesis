#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:43:09 2022

@author: frederikhartmann
"""

import numpy as np
import os
from torch.utils.data import Dataset
import glob
from PIL import Image

class BacteriaDataset(Dataset):
    # Initialize
    def __init__(self, image_dir, mask_dir, transform=None):
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
        image = np.array(Image.open(img_path).convert("L"))

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
