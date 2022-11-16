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
from metrics import numeric_score, precision_score, recall_score, specificity_score
from metrics import intersection_over_union, accuracy_score, dice_score, jaccard_score

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
    dice = []
    ACC = []
    Spec = []
    Prec = []
    Recall = []
    softmax = torch.nn.Softmax(dim=1)
    # put model in eval
    model.eval()
    loss_fn=nn.CrossEntropyLoss()
    running_loss=0
    # Go through loader
    with torch.no_grad():
        for x,y in loader:
            # Send to gpu or cpu
            x = x.float().to(device=device)
            y = y.float().to(device=device).permute(0,3,1,2)

            # Predict, permute, softmax and argmax
            preds = model(x)
            loss = loss_fn(preds, y)
            running_loss+=loss.item()

            # Softmax and argmax to get predictions
            preds = softmax(preds)
            preds = torch.argmax(preds, dim=1)

            # compute metrics
            FP, FN, TP, TN = numeric_score(preds, y)
            dice.append(dice_score(preds, y))
            ACC.append(accuracy_score(FP, FN, TP, TN))
            Spec.append(specificity_score(FP, FN, TP, TN))
            Prec.append(precision_score(FP, FN, TP, TN))
            Recall.append(recall_score(FP, FN, TP, TN))

    # Convert metrics to numpy arrays for averaging
    dice = np.array(dice)
    ACC = np.array(ACC)
    Spec = np.array(Spec)
    Prec = np.array(Prec)
    Recall = np.array(Recall)

    # Get number of elements in loader to average with    
    N=len(loader)
    
    # Set up metrics
    Metrics={"dice_score":dice.sum(axis=0)/N,"Accuracy":ACC.sum(axis=0)/N,"Specificity":Spec.sum(axis=0)/N,"Precision":Prec.sum(axis=0)/N,"Recall":Recall.sum(axis=0)/N}
            
    # Print statements
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