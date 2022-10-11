#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:07:32 2022

@author: frederikhartmann
"""
from model import weights_init
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import Unet
from utils import (load_checkpoint,
                    save_checkpoint,
                    get_loaders,
                    check_accuracy,
                    save_predictions_as_imgs,
                    IoULoss,
                    )
from torch.utils.tensorboard import SummaryWriter

# Create writerr for tensorboard
writer = SummaryWriter("./reports/figures/tensorboard")

# Hyper parameters
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1
NUM_EPOCHS = 10
NUM_WORKERS = 1

# CROP 
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

PIN_MEMORY = False
LOAD_MODEL = False
TRAIN_IMG_DIR = "/work3/s174182/data/external/stage1_train"
TRAIN_MASK_DIR = "/work3/s174182/data/external/stage1_train"
VAL_IMG_DIR = "/work3/s174182/data/external/stage1_val"
VAL_MASK_DIR = "/work3/s174182/data/external/stage1_val"


# Train function does one epoch
def train_fn(loader, model, optimizer, loss_fn, scaler, scheduler):
    loop = tqdm(loader)
    print("Training on: ",DEVICE)
    # Go through batch
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.float().to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward pass
        # with torch.cuda.amp.autocast():
        predictions = model(data)
        loss = loss_fn(predictions, targets)
            
        # Backward probagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ##### SCALER #####

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        # Update tqdm 
        loop.set_postfix(loss=loss.item())
        
        # Update running loss
        running_loss += loss.item()
        
    return loss
  
def main():
    # Transformation on train set
    train_transform = A.Compose([
        A.augmentations.geometric.transforms.HorizontalFlip(p=0.5),
        A.augmentations.geometric.transforms.VerticalFlip(p=0.5),
        A.augmentations.geometric.rotate.Rotate(limit=180, interpolation=1, border_mode=4, value=None, mask_value=None, rotate_method='largest_box', crop_border=False, always_apply=False, p=0.5),
        ToTensorV2(),
        ])
    
    # Validation transforms
    val_transform = A.Compose([
        A.augmentations.geometric.transforms.HorizontalFlip(p=0.5),
        A.augmentations.geometric.transforms.VerticalFlip(p=0.5),
        A.augmentations.geometric.rotate.Rotate(limit=180, interpolation=1, border_mode=4, value=None, mask_value=None, rotate_method='largest_box', crop_border=False, always_apply=False, p=0.5),
        ToTensorV2(),
        ])
    
    # Create model, loss function, optimizer, loaders, scaler
    model = Unet(in_channels = 1, out_channels = 1).to(device=DEVICE)
    model.apply(weights_init)
    
    # Add other parameters to log
    writer.add_scalar("Batch size", BATCH_SIZE)
    writer.add_scalar("Learning rate", LEARNING_RATE)
    
    # If we load model, load the checkpoint
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth"), model)
    

    loss_fn = IoULoss() #nn.BCEWithLogitsLoss() # For flere klasse, ændr til CELoss

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-8) # add more params if wanted
    gamma = 0.9
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)# Learning rate scheduler
    
    ###########################################
    # optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss()
    # global_step = 0
    ###########################################


    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        num_workers=NUM_WORKERS,
      )
    writer.add_graph(model, iter(train_loader).next()[0].to(device=DEVICE))

    scaler = torch.cuda.amp.GradScaler()
    best_score=0
    # Go through epochs
    for epoch in range(NUM_EPOCHS):
        loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, scheduler)
        
                          
        # check accuracy
        dice_score=check_accuracy(val_loader, model, device=DEVICE)
        
        # Log loss and dice score
        writer.add_scalar("Training loss", loss, epoch)
        writer.add_scalar("Running score", dice_score, epoch)
        
        if dice_score>best_score:
            # Save model, check accuracy, print some examples to folder
            checkpoint = {"state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict(),}
            save_checkpoint(checkpoint)
            best_score=dice_score
        
        # Save images
        # save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)
    
    writer.close()
    
    pass

if __name__ == "__main__":
    main()
