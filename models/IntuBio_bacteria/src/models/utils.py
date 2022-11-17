#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:37:52 2022

@author: frederikhartmann
"""
from asyncio.constants import ACCEPT_RETRY_DELAY
import torch
import torch.nn as nn
import torchvision
import sys
import os
import numpy as np
sys.path.append('src/data')
from make_dataset import BacteriaDataset

from torch.utils.data import DataLoader

# save checkpoint
def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)
    
# Load checkpoint
def load_checkpoint(checkpoint, model):
    print("=> Saving Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
# Get loaders
def get_loaders(train_dir, train_mask_dir, val_dir, val_mask_dir, 
                batch_size, train_transform, val_transform, num_workers=1, pin_memory=True):
    
    # Train dataloader
    train_ds = BacteriaDataset(image_dir = train_dir, mask_dir=train_mask_dir, transform = train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    
    # Validation dataloader
    valid_ds = BacteriaDataset(image_dir = val_dir, mask_dir=val_mask_dir, transform = val_transform)
    valid_loader = DataLoader(valid_ds, batch_size=1, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    
    return train_loader, valid_loader

# Check accuracy

def check_accuracy(loader, model, device="cpu"):
    # init
    num_correct = 0
    num_pixels = 0
    dice = 0
    ACC=0
    Spec=0
    Prec=0
    Recall=0
    # put model in eval
    model.eval()
    loss_fn=IoULoss()
    loss_fn2=nn.BCEWithLogitsLoss()
    running_loss=0
    # Go through loader
    with torch.no_grad():
        for x,y in loader:
            # Send to gpu or cpu
            x = x.float().to(device=device)
            y = y.float().unsqueeze(1).to(device=device)
    
            # Predict (sigmoid good for binary)
            preds = model(x)
            loss = loss_fn(preds,y)+1/4*loss_fn2(preds,y)
            running_loss+=loss.item()

            preds = (preds>0.5).float()
            # count
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            TP = torch.sum(torch.logical_and(preds == 1.0, y == 1.0))
            TN = torch.sum(torch.logical_and(preds == 0.0, y == 0.0))
            FN = torch.sum(torch.logical_and(preds == 0.0, y == 1.0))   
            FP = torch.sum(torch.logical_and(preds == 1.0, y == 0.0))

            dice += 2 * (TP)/(FP+2*TP+FN+1e-8)
            ACC += (TP+TN)/(TP+TN+FP+FN)
            Spec += (TN/(FP+TN+1e-8))
            Prec += TP/(TP+FP+1e-8)
            Recall+= TP/(TP+FN+1e-8)
    
    N=len(loader)
    Metrics={"dice_score":dice/N,"Accuracy":ACC/N,"Specificity":Spec/N,"Precision":Prec/N,"Recall":Recall/N}
            
    
    # Print statements
    print(len(loader))
    print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}")
    print("Metrics",Metrics)
    
    # Set model back to training
    model.train()
    
    return Metrics, running_loss/len(loader)
    
    
# Save predictions
    
def save_predictions_as_imgs(loader, model, folder = "saved_images/", device="cpu"):
    # Set model to evaluation
    model.eval()
    
    # Go through loader
    for idx, (x,y) in enumerate(loader):
        # set to device
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            # Predict
            preds = torch.sigmoid(x)
            preds = (preds > 0.5).float()
            
            # save
            torchvision.utils.save_image(preds, f"{folder}pred_{idx}.png")
            
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}target_{idx}.png")
    
    model.train()


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

def recon_im(patches: np.ndarray, im_h: int, im_w: int, n_channels: int, stride: int):
    """Reconstruct the image from all patches.
        Patches are assumed to be square and overlapping depending on the stride. The image is constructed
         by filling in the patches from left to right, top to bottom, averaging the overlapping parts.
    Parameters
    -----------
    patches: 4D ndarray with shape (patch_number,patch_height,patch_width,channels)
        Array containing extracted patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.
    im_h: int
        original height of image to be reconstructed
    im_w: int
        original width of image to be reconstructed
    n_channels: int
        number of channels the image has. For  RGB image, n_channels = 3
    stride: int
           desired patch stride
    Returns
    -----------
    reconstructedim: ndarray with shape (height, width, channels)
                      or ndarray with shape (height, width) if output image only has one channel
                    Reconstructed image from the given patches
    """

    patch_size = patches.shape[1]  # patches assumed to be square

    # Assign output image shape based on patch sizes
    rows = ((im_h - patch_size) // stride) * stride + patch_size
    cols = ((im_w - patch_size) // stride) * stride + patch_size

    if n_channels == 1:
        reconim = np.zeros((rows, cols))
        divim = np.zeros((rows, cols))
    else:
        reconim = np.zeros((rows, cols, n_channels))
        divim = np.zeros((rows, cols, n_channels))

    p_c = (cols - patch_size + stride) / stride  # number of patches needed to fill out a row

    totpatches = patches.shape[0]
    initr, initc = 0, 0

    # extract each patch and place in the zero matrix and sum it with existing pixel values

    reconim[initr:patch_size, initc:patch_size] = patches[0]# fill out top left corner using first patch
    divim[initr:patch_size, initc:patch_size] = np.ones(patches[0].shape)

    patch_num = 1

    while patch_num <= totpatches - 1:
        initc = initc + stride
        reconim[initr:initr + patch_size, initc:patch_size + initc] += patches[patch_num]
        divim[initr:initr + patch_size, initc:patch_size + initc] += np.ones(patches[patch_num].shape)

        if np.remainder(patch_num + 1, p_c) == 0 and patch_num < totpatches - 1:
            initr = initr + stride
            initc = 0
            reconim[initr:initr + patch_size, initc:patch_size] += patches[patch_num + 1]
            divim[initr:initr + patch_size, initc:patch_size] += np.ones(patches[patch_num].shape)
            patch_num += 1
        patch_num += 1
    # Average out pixel values
    reconstructedim = reconim / divim

    return reconstructedim

def get_patches(GT, stride, patch_size):
    """Extracts square patches from an image of any size.
    Parameters
    -----------
    GT : ndarray
        n-dimensional array containing the image from which patches are to be extracted
    stride : int
           desired patch stride
    patch_size : int
               patch size
    Returns
    -----------
    patches: ndarray
            array containing all patches
    im_h: int
        height of image to be reconstructed
    im_w: int
        width of image to be reconstructed
    n_channels: int
        number of channels the image has. For  RGB image, n_channels = 3
    """

    hr_patches = []

    for i in range(0, GT.shape[0] - patch_size + 1, stride):
        for j in range(0, GT.shape[1] - patch_size + 1, stride):
            hr_patches.append(GT[i:i + patch_size, j:j + patch_size])

    im_h, im_w = GT.shape[0], GT.shape[1]

    if len(GT.shape) == 2:
        n_channels = 1
    else:
        n_channels = GT.shape[2]

    patches = np.asarray(hr_patches)

    return patches, im_h, im_w, n_channels