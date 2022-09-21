#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:37:52 2022

@author: frederikhartmann
"""
import torch
import torchvision
from dataset import BacteriaDataset
from torch.utils.data import DataLoader

# save checkpoint
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
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
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    
    return train_loader, valid_loader

# Check accuracy

def check_accuracy(loader, model, device="cpu"):
    # init
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    # put model in eval
    model.eval()
    
    # Go through loader
    with torch.no_grad():
        for x,y in loader:
            # Send to gpu or cpu
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
    
            # Predict (sigmoid good for binary)
            preds = model(x)
            preds = (preds>0.5).float()
            
            # count
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds*y).sum())/((preds+y).sum() + 1e-8)
    
    # Print statements
    print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    
    # Set model back to training
    model.train()
    
    return dice_score/len(loader)
    
    
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
    
    
    