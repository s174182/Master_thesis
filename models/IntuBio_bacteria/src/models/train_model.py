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
                    FocalLoss,
                    )
import os
import hydra
from hydra import compose, initialize
from datetime import datetime
import yaml
import wandb

# Hyper parameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELNAME= str(datetime.now())+".pth"

PIN_MEMORY = False
LOAD_MODEL = False

Debug_MODE=True
if Debug_MODE:
        TRAIN_IMG_DIR = "/work3/s174182/debug/Annotated_segmentation_patch/train/"
        TRAIN_MASK_DIR = "/work3/s174182/debug/Annotated_segmentation_patch/train/"
        VAL_IMG_DIR = "/work3/s174182/debug/Annotated_segmentation_patch/val/"
        VAL_MASK_DIR = "/work3/s174182/debug/Annotated_segmentation_patch/val/"
else:
    TRAIN_IMG_DIR = "/work3/s174182/train_data/Annotated_segmentation_patch_no_empty_masks/train/"
    TRAIN_MASK_DIR = "/work3/s174182/train_data/Annotated_segmentation_patch_no_empty_masks/train/"
    VAL_IMG_DIR = "/work3/s174182/train_data/Annotated_segmentation_patch_no_empty_masks/val/"
    VAL_MASK_DIR = "/work3/s174182/train_data/Annotated_segmentation_patch_no_empty_masks/val/"


# Train function does one epoch
def train_fn(loader, model, optimizer, loss_fn, loss_fn2,wloss_1,wloss_2):
    loop = tqdm(loader,position=0,leave=True)
    # Go through batch
    running_loss = 0.0
    with loop as pbar:
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.float().to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)

            # Backward probagation  
            optimizer.zero_grad()

            # Forward pass
            # with torch.cuda.amp.autocast():
            predictions = model(data)
            print(predictions.shape)
            print(targets.shape)

            loss = wloss_1*loss_fn(predictions, targets)+wloss_2*loss_fn2(predictions, targets)

                
            # Backward propagate and step optimizer
            loss.backward()
            optimizer.step()
            
            
            
            # Update tqdm 
            pbar.set_postfix(loss=loss.item())
            pbar.update()
            # Update running loss
            running_loss += loss.item()
        
    return running_loss/len(loop)

def main():
    # Load sweep configuration
    with open('basic.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    # Initialize W and B
    run = wandb.init(project='{MODELNAME}'.replace(".", "_").replace(":","_"),entity="intubio",config=config)

    LEARNING_RATE = config['hyperparameters']['learning_rate']
    BATCH_SIZE = config['hyperparameters']['batch_size']
    WEIGHT_DECAY = config['hyperparameters']['weight_decay']
    OPTIMIZER = config['hyperparameters']['optimizer']
    NUM_EPOCHS = config['hyperparameters']['num_epochs']
    NUM_WORKERS = config['hyperparameters']['num_workers']
    wandb.config.update({
    "dataset" : TRAIN_IMG_DIR
    })
    wloss_1=config["hyperparameters"]["w_loss1"]
    wloss_2=config["hyperparameters"]["w_loss2"]
    loss_fn2 = IoULoss()
    loss_fn = nn.BCEWithLogitsLoss()
   
    #Transformation on train set
    train_transform = A.Compose([
        A.augmentations.geometric.transforms.HorizontalFlip(p=0.5),
        A.augmentations.geometric.transforms.VerticalFlip(p=0.5),
        A.augmentations.geometric.rotate.Rotate(limit=180, interpolation=1, border_mode=4, value=None, mask_value=None, rotate_method='largest_box', crop_border=False, always_apply=False, p=0.5),
        A.augmentations.transforms.Normalize (mean=(145.7), std=(46.27), max_pixel_value=1.0, always_apply=False, p=1.0),
        ToTensorV2(),
        ])
    
    # Validation transforms
    val_transform = A.Compose([
        A.augmentations.geometric.transforms.HorizontalFlip(p=0.5),
        A.augmentations.geometric.transforms.VerticalFlip(p=0.5),
        A.augmentations.geometric.rotate.Rotate(limit=180, interpolation=1, border_mode=4, value=None, mask_value=None, rotate_method='largest_box', crop_border=False, always_apply=False, p=0.5),
        A.augmentations.transforms.Normalize (mean=(145.7), std=(46.27), max_pixel_value=1.0, always_apply=False, p=1.0),
        ToTensorV2(),
        ])

    
    # Create model, loss function, optimizer, loaders
    model = Unet(in_channels = 1, out_channels = 1).to(device=DEVICE)
    
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

    best_score=0
    # Go through epochs
    for epoch in range(NUM_EPOCHS):
        print("Training epoch:", epoch)
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, loss_fn2,wloss_1,wloss_2)         
        # check accuracy
        Metrics, val_loss=check_accuracy(val_loader, model, device=DEVICE)


        wandb.log({'training_loss': train_loss})
        wandb.log({'validation_loss': val_loss})
        wandb.log({'dice_score': Metrics["dice_score"]})
        wandb.log({'Accuracy': Metrics["Accuracy"]})
        wandb.log({'Specificity': Metrics["Specificity"]})
        wandb.log({'Precision': Metrics["Precision"]})
        wandb.log({'Recall': Metrics["Recall"]})
        wandb.log({'epoch': epoch})




        if Metrics["dice_score"]>best_score:
            # Save model, check accuracy, print some examples to folder
            checkpoint = {"state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict(),}
            save_checkpoint(checkpoint,MODELNAME)
            best_score=Metrics["dice_score"]
        
        
    
    pass

if __name__ == "__main__":
    main()
    
