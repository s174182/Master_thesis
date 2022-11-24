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

PIN_MEMORY = False
LOAD_MODEL = False

Debug_MODE=False
if Debug_MODE:
        TRAIN_IMG_DIR = "/work3/s174182/debug/RGB_method/train/"
        TRAIN_MASK_DIR = "/work3/s174182/debug/RGB_method/train/"
        VAL_IMG_DIR = "/work3/s174182/debug/RGB_method/val/"
        VAL_MASK_DIR = "/work3/s174182/debug/RGB_method/val/"
else:
    TRAIN_IMG_DIR = "/work3/s174182/train_data/RGB_method_balanced_1/train/"
    TRAIN_MASK_DIR = "/work3/s174182/train_data/RGB_method_balanced_1/train/"
    VAL_IMG_DIR = "/work3/s174182/train_data/RGB_method_balanced_1/val/"
    VAL_MASK_DIR = "/work3/s174182/train_data/RGB_method_balanced_1/val/"


# Train function does one epoch
def train_fn(loader, model, optimizer, loss_fn, loss_fn2, w_loss1, w_loss2):
    """
    Function to train a model on specified training data
    and validating on specified validation data
    args: 
        loader: DataLoader created using the make_dataset.py file holding training data
                in batches, together with transforms and more.
        model: The loaded (UNet) model that is to be trained
        optimizer: Optimizer used to compute gradients used for loss and model training
        loss_fn: Loss function 1
        loss_fn: Loss function 2
        w_loss1: Weighting of loss function 1
        w_loss2: Weighting of loss function 2

    Output:
        running_loss: Computed loss in the current epoch
        updated weights in network
    """
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
            loss = w_loss1*loss_fn(predictions, targets) + w_loss2*loss_fn2(predictions, targets)
                
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
    """
    Function for book-keeping the training cycle and validation on validation set. 
    This also specifies transformations used on data (augmentations)
    Runs Weights and Biases logging for visualization purposes
    
    Needs basic.yaml file as configuration for training, holding the following fixed values
        - learning_rate: The learning rate used in optimizer
        - batch_size: Batch size used in training cycle
        - weight_decay: Regularization parameter in optimizer
        - optimizer: What optimizer should be used (currently either sgd or adam)
        - num_epochs: Number of epochs needed for training
        - num_workers: Number of workers needed for training, default should be 1
        - w_loss1: Weighting of loss function 1
        - w_loss2: Weighting of loss function 2

    Output:
        Model state dictionary for model with best score
        Weights and Biases logs on their website wandb.ai 
        NB: 
            To run with wandb create a profile on wandb.ai, 
            make an entity (team) on their website
            and input the name in wandb.init. For help, see wandb documentation on
            https://docs.wandb.ai/

    """
    
    # Load config
    with open('basic.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    wandb.init(entity='intubio', config=config)

    # Set configuration hyperparameters
    LEARNING_RATE = config['hyperparameters']['learning_rate']
    BATCH_SIZE = config['hyperparameters']['batch_size']
    WEIGHT_DECAY = config['hyperparameters']['weight_decay']
    OPTIMIZER = config['hyperparameters']['optimizer']
    NUM_EPOCHS = config['hyperparameters']['num_epochs']
    NUM_WORKERS = config['hyperparameters']['num_workers']
    loss_fn = nn.BCEWithLogitsLoss() # Binary cross entropy
    loss_fn2 = IoULoss() # IoU loss

    # Weight of loss functions
    w_loss1 = config['hyperparameters']['w_loss1']
    w_loss2 = config['hyperparameters']['w_loss2']

    # Skipborders parameter
    SKIPBORDERS = False

    #Transformation on train set
    # Mean and std can be calculated in mean_std found in subfolder data
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
        load_checkpoint(torch.load("2022-10-25 12:40:16.459699.pth"), model)

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
        skipborders=SKIPBORDERS,
      )
    
#    writer.add_graph(model, iter(train_loader).next()[0].to(device=DEVICE))

    #scaler = torch.cuda.amp.GradScaler()
    best_score=0
    # Go through epochs
    for epoch in range(NUM_EPOCHS):
        print("Training epoch:", epoch)
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, loss_fn2, w_loss1, w_loss2)         
        # check accuracy
        dice_score, val_loss=check_accuracy(val_loader, model, device=DEVICE)
        
        wandb.log({'training_loss': train_loss})
        wandb.log({'validation_loss': val_loss})
        wandb.log({'dice_score': Metrics["dice_score"][1]})
        wandb.log({'Accuracy': Metrics["Accuracy"][1]})
        wandb.log({'Specificity': Metrics["Specificity"][1]})
        wandb.log({'Precision': Metrics["Precision"][1]})
        wandb.log({'Recall': Metrics["Recall"][1]})
        wandb.log({'epoch': epoch})

        print(Metrics["dice_score"][1])


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
    
