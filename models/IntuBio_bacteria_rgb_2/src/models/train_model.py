#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:07:32 2022

@author: frederikhartmann
"""
import argparse
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
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
import os
import hydra
from datetime import datetime
import yaml
import wandb

# Hyper parameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELNAME= str(datetime.now())+".pth"


# Initialize W and B
wandb.init(project='{MODELNAME}'.replace(".", "_").replace(":","_"), entity="intubio")


PIN_MEMORY = False
LOAD_MODEL = False

Debug_MODE=False
if Debug_MODE:
        TRAIN_IMG_DIR = "/work3/s174182/debug/RGB_method/train/"
        TRAIN_MASK_DIR = "/work3/s174182/debug/RGB_method/train/"
        VAL_IMG_DIR = "/work3/s174182/debug/RGB_method/val/"
        VAL_MASK_DIR = "/work3/s174182/debug/RGB_method/val/"
else:
    TRAIN_IMG_DIR = "/work3/s174182/train_data/RGB_method_balanced/train/"
    TRAIN_MASK_DIR = "/work3/s174182/train_data/RGB_method_balanced/train/"
    VAL_IMG_DIR = "/work3/s174182/train_data/RGB_method_balanced/val/"
    VAL_MASK_DIR = "/work3/s174182/train_data/RGB_method_balanced/val/"


# Train function does one epoch
def train_fn(loader, model, optimizer, loss_fn,loss_fn2, scaler):
    loop = tqdm(loader,position=0,leave=True)
    # Go through batch
    running_loss = 0.0
    with loop as pbar:
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.float().to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)

            # Forward pass
            # with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)+1/4*loss_fn2(predictions, targets)
                
            # Backward probagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ##### SCALER #####

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            
            # Update tqdm 
            pbar.set_postfix(loss=loss.item())
            pbar.update()
            # Update running loss
            running_loss += loss.item()
        
    return running_loss/len(loop)

def main():
    # Load sweep configuration
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    # Initialize W and B
    run = wandb.init(allow_val_change=True,entity="intubio",config = config) #project='{MODELNAME}'.replace(".", "_").replace(":","_"), entity="intubio", 

    # Set configuration hyperparameters
    LEARNING_RATE = wandb.config.lr
    BATCH_SIZE = wandb.config.batch_size
    WEIGHT_DECAY = wandb.config.wd
    OPTIMIZER = wandb.config.optimizer
    NUM_EPOCHS = 10
    NUM_WORKERS = 1
    loss_fn = IoULoss()#IoULoss()# if cfg.hyperparameters.lossfn=="IoU" else nn.BCEWithLogitsLoss() # For flere klasse, ændr til CELoss
    loss_fn2 =nn.BCEWithLogitsLoss()

    #Transformation on train set
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
    model = Unet(in_channels = 3, out_channels = 1).to(device=DEVICE)
    #model.apply(weights_init)
    
    # Set wandb to watch the model
    wandb.watch(model, log_freq=100)

    # If we load model, load the checkpoint
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth"), model)

    #writer.add_scalar("Loss function", "IoULoss")
    if OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # add more params if wanted
    elif OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # add more params if wanted

    #gamma = 0.9
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)# Learning rate scheduler
    
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
    
#    writer.add_graph(model, iter(train_loader).next()[0].to(device=DEVICE))

    scaler = torch.cuda.amp.GradScaler()
    best_score=0
    # Go through epochs
    for epoch in range(NUM_EPOCHS):
        print("Training epoch:", epoch)
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, loss_fn2, scaler)         
        # check accuracy
        dice_score, val_loss=check_accuracy(val_loader, model, device=DEVICE)
        
        wandb.log({'training_loss': train_loss})
        wandb.log({'validation_loss': val_loss})
        wandb.log({'dice_score': dice_score})
        wandb.log({'epoch': epoch})


        if dice_score>best_score:
            # Save model, check accuracy, print some examples to folder
            checkpoint = {"state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict(),}
            save_checkpoint(checkpoint,MODELNAME)
            best_score=dice_score
        
        # Save images
        # save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)
    
    
    pass

if __name__ == "__main__":
    main()
    
