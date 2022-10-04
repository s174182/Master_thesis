#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:43:09 2022

@author: frederikhartmann
"""

import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

class BacteriaDataset(Dataset):
    # Initialize
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = list(set(os.listdir(image_dir)) - {'desktop.ini'})

    # Define length of dataset
    def __len__(self):
        return len(self.images)


    # Define get item for data loader
    def __getitem__(self, index):
        # get image
        img_path = os.path.join(self.image_dir, self.images[index]+"/images/"+self.images[index]+".png")
        image = np.array(Image.open(img_path).convert("L"))

        mask_path = os.path.join(self.image_dir, self.images[index]+"/masks/")
        masks = list(set(os.listdir(mask_path)) - {'desktop.ini'})

        merged_masks=np.zeros((image.shape))
        for mask in masks:
            temp_masks=np.array(Image.open(mask_path+"/"+mask).convert("L"), dtype=np.float32)
            merged_masks+=temp_masks

        # img_path = os.path.join(self.image_dir, self.images[index])

        # # Get mask
        # mask_path = os.path.join(self.mask_dir, self.images[index].replace("TotalProjI", "Target"))

        # Load image and mask

        mask = merged_masks #np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

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
