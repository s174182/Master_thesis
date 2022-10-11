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
from torch.utils.tensorboard import SummaryWriter
import os
import hydra
from datetime import datetime
import yaml

# Create writer for tensorboard
writer = SummaryWriter("./reports/figures/tensorboard")

# Hyper parameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELNAME= str(datetime.now())+".pth"
#Add model name to yaml file and create a copy
stream = open("basic.yaml", 'r')
my_dict = yaml.full_load(stream)
cpy = my_dict
cpy["modelname"]=MODELNAME
with open(f'{MODELNAME}'.replace(".", "_").replace(":","_")+".yaml", 'w') as file:
    documents = yaml.dump(cpy, file)

writer.add_text('model name', MODELNAME)



PIN_MEMORY = False
LOAD_MODEL = False

Debug_MODE=False
if Debug_MODE:
        TRAIN_IMG_DIR = "/work3/s174182/debug/Annotated_segmentation_patch/train/"
        TRAIN_MASK_DIR = "/work3/s174182/debug/Annotated_segmentation_patch/train/"
        VAL_IMG_DIR = "/work3/s174182/debug/Annotated_segmentation_patch/val/"
        VAL_MASK_DIR = "/work3/s174182/debug/Annotated_segmentation_patch/val/"
else:
    TRAIN_IMG_DIR = "/work3/s174182/train_data/Annotated_segmentation_patch/train/"
    TRAIN_MASK_DIR = "/work3/s174182/train_data/Annotated_segmentation_patch/train/"
    VAL_IMG_DIR = "/work3/s174182/train_data/Annotated_segmentation_patch/val/"
    VAL_MASK_DIR = "/work3/s174182/train_data/Annotated_segmentation_patch/val/"


# Train function does one epoch
def train_fn(loader, model, optimizer, loss_fn,loss_fn2, scaler, scheduler):
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
print(os.getcwd())
@hydra.main(version_base="1.2", config_path="../../",config_name="basic.yaml")
def main(cfg):
    #Hyperparameters
    LEARNING_RATE = cfg.hyperparameters.learning_rate
    BATCH_SIZE = cfg.hyperparameters.batch_size
    NUM_EPOCHS = cfg.hyperparameters.num_epochs
    NUM_WORKERS = cfg.hyperparameters.num_workers
    loss_fn = IoULoss()# if cfg.hyperparameters.lossfn=="IoU" else nn.BCEWithLogitsLoss() # For flere klasse, ændr til CELoss
    loss_fn2 =nn.BCEWithLogitsLoss()
    WEIGHT_DECAY=cfg.hyperparameters.weight_decay
    
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
    model = Unet(in_channels = 1, out_channels = 1).to(device=DEVICE)
    #model.apply(weights_init)
    
    # Add other parameters to log
    writer.add_scalar("Batch size", BATCH_SIZE)
    writer.add_scalar("Learning rate", LEARNING_RATE)
    
    # If we load model, load the checkpoint
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth"), model)
    
    
    #writer.add_scalar("Loss function", "IoULoss")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # add more params if wanted
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
    
#    writer.add_graph(model, iter(train_loader).next()[0].to(device=DEVICE))

    scaler = torch.cuda.amp.GradScaler()
    best_score=0
    # Go through epochs
    for epoch in range(NUM_EPOCHS):
        print("Training epoch:", epoch)
        train_loss = train_fn(train_loader, model, optimizer, loss_fn,loss_fn2, scaler, scheduler)         
        # check accuracy
        dice_score, val_loss=check_accuracy(val_loader, model, device=DEVICE)
        
        # Log loss and dice score
        writer.add_scalar("Training loss", train_loss, epoch)
        writer.add_scalar("Validation loss", val_loss, epoch)
        writer.add_scalar("Validation Dice_score", dice_score, epoch)
        
        if dice_score>best_score:
            # Save model, check accuracy, print some examples to folder
            checkpoint = {"state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict(),}
            save_checkpoint(checkpoint,MODELNAME)
            best_score=dice_score
        
        # Save images
        # save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)
    
    writer.close()
    
    pass

if __name__ == "__main__":
    main()
    
