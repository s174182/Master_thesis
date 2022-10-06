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
                self.images.append(f + "/" + sf + "/mask/Mask_1.png")
                self.masks.append(f + "/" + sf + "/img/INorm.png")
        
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
        
        # print(image.shape)
        # print(mask.shape)
        # cv2.imshow('img', image.numpy().squeeze())
        # cv2.imshow("mask",mask.numpy())
        # cv2.waitKey()
        return image, mask
