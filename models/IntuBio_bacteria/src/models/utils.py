#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:37:52 2022

@author: frederikhartmann
"""
import torch
import torch.nn as nn
import torchvision
import sys
import os
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
    dice_score = 0
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
            dice_score += (2 * (preds*y).sum())/((preds+y).sum() + 1e-8)
    
    # Print statements
    print(len(loader))
    print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    
    # Set model back to training
    model.train()
    
    return dice_score/len(loader), running_loss/len(loader)
    
    
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