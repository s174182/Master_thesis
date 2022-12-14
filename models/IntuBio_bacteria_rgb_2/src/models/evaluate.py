# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:32:58 2022

@author: Jonat

This script is meant for evalutation, it loads a full Image together with its mask.
Then it pads the image to preserve the full image, crop it into patches,
predicts on those stitch them back together and measure the dice score and other metrics.

"""

import cv2
import numpy as np
import os
import argparse
from patchify import patchify, unpatchify
import torch
import torchvision.transforms as transforms
from model import Unet
from scipy import ndimage
import matplotlib.pyplot as plt
import json
import sys
from PIL import Image
sys.path.append('../data/')
from utils import load_checkpoint, recon_im
import albumentations as A
import seaborn as sn

def evaluate(img_path,mask_path,model,DEVICE='cpu', threshold=0.5, step=256, downscale=512):
    """
    Function to evaluate a trained model on an independent test set
    args: 
        img_path: Path to test-images given as string
        mask_path: Path to test-masks given as string
        model: UNet model with loaded state-dictionary from a trained model
        device: torch.device('cuda' if torch.cuda.is_available() else 'cpu') - GPU if available
                else cpu
        threshold: Threshold in probability map - defaults to 0.5
        step: Stepsize in patches generated - defaults to 512//2
        downscale: Patch sizes are of dimension DOWNSCALE x DOWNSCALE

    Output:
        Metrics: Dictionary of metrics
            - Dice score
            - Accuracy
            - Precision
            - Recall
            - Specificity
        Saved images: Saves probability map image and prediction image in specified path

    """
    print(img_path)
    transform = transforms.ToTensor()
    model.eval()

    DOWNSCALE=downscale # this is equal to the patch size
    
    # Read image and convert from BGR to RGB
    img_orig= np.array(Image.open(img_path))
    N,M,_=np.shape(img_orig)

    # Mask
    mask = cv2.imread(mask_path, 0)
    mask[mask>1]=1
    
    #Pad the image, nessecary in order to crop the image into 512X512 patches
    hor_pad=np.ceil(img_orig.shape[1]/DOWNSCALE)*DOWNSCALE #padding in horizontal direction
    ver_pad=np.ceil(img_orig.shape[0]/DOWNSCALE)*DOWNSCALE #padding in vertical direction
    img=np.zeros((int(ver_pad),int(hor_pad),3))
    img[0:img_orig.shape[0],0:img_orig.shape[1],:]=img_orig
    

    # Create the patches
    patches=patchify(img,(DOWNSCALE,DOWNSCALE,3),step=step)

    # Loop through and predict each patch
    pred_patches=[]
    with torch.no_grad():
        for i in range(patches.shape[0]):
            for j in range (patches.shape[1]):
                # Transform image to tensor
                x = patches[i, j, 0, :, :, :]                

                # Normalize image
                im_mean1, im_std1 = np.mean(x[:,:,0]), np.std(x[:,:,0])
                im_mean2, im_std2 = np.mean(x[:,:,1]), np.std(x[:,:,1])
                im_mean3, im_std3 = np.mean(x[:,:,2]), np.std(x[:,:,2])
                
                # Normalize image as to match training
                if np.sum(x) > 1:
                    pred_transform = A.Compose([A.augmentations.transforms.Normalize(mean=(im_mean1, im_mean2, im_mean3), std=(im_std1, im_std2, im_std3), max_pixel_value=1, always_apply=True, p=1.0),
                    ])
                    augmented = pred_transform(image=x)
                    x=augmented["image"]

                # To tensor
                x = transform(x)

                # Predict
                x = x[None,:].float().to(device=DEVICE) #as its not a batch do a dummy expansion
                preds = model(x)
                preds = torch.sigmoid(preds)
                pred_patches.append(preds)

    pred_patches = np.array([pred_patches[k].cpu().numpy().squeeze() for k in range(0,len(pred_patches))])
    pred_mask =recon_im(pred_patches, img.shape[0], img.shape[1],1,step)
    pred_mask = (pred_mask[:img_orig.shape[0],:img_orig.shape[1]])
    heatmap_copy=pred_mask
    pred_mask= (pred_mask>threshold).astype(np.uint8)
    
    #Metrics
    TP = np.sum(np.logical_and(pred_mask == 1, mask == 1))
    TN = np.sum(np.logical_and(pred_mask == 0, mask == 0))
    FN = np.sum(np.logical_and(pred_mask == 0, mask == 1))   
    FP = np.sum(np.logical_and(pred_mask == 1, mask == 0))

    # Computer dice scores, accuracy, specificity, precision, recall
    dice = 2 * (TP)/(FP+2*TP+FN+1e-8)
    ACC = (TP+TN)/(TP+TN+FP+FN)
    Spec = (TN/(FP+TN+1e-8))
    Prec = TP/(TP+FP+1e-8)
    Recall= TP/(TP+FN+1e-8)
    
    # Save metrics in dictionary 
    Metrics={"dice":dice,"Accuracy":ACC,"Specificity":Spec,"Precision":Prec,"Recall":Recall}

    #### SAVE IMAGE ####
    save_folder="_".join(img_path.split("/")[-3:-1])
    img_merged = cv2.merge((img_orig[:,:,0],img_orig[:,:,0],img_orig[:,:,0]))
    shapes=np.zeros((N,M,3)).astype(np.uint8)
    shapes[pred_mask==1,0]=255;shapes[pred_mask==1,1]=255; # create a filter for the predicted masks
    predicted_img = cv2.addWeighted(img_merged,1,shapes,0.25,0.1)
   
   # Create directory
    os.makedirs("/work3/s174182/predictions/"+model_name,exist_ok=True)
   
    cv2.imwrite("/work3/s174182/predictions/"+model_name+"/"+save_folder+".png",predicted_img)
    plt.ioff()
    svm = sn.heatmap(heatmap_copy,yticklabels=False,xticklabels=False)
    figure = svm.get_figure()    
    figure.savefig("/work3/s174182/predictions/"+model_name+"/"+save_folder+"prob_map"+".png", dpi=1555)
    plt.close()

    return Metrics

# Paths and model names
test_path = "/work3/s174182/Test_data/"
model_name = "supernatural-possession-43" #azure-spaceship-44"
model_path = "../../models/"+model_name+".pth" #to be made as an argument
os.makedirs("../../data/predictions/"+model_name,exist_ok=True) # to save predicted images in

# Create model, and load state dictionary from the model defined in "model_name"
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(in_channels=3, out_channels=1)
checkpoint=torch.load(model_path)  
model.to(device=DEVICE)
load_checkpoint(checkpoint,model) 

# Set stepsize and threshold
DOWNSCALE=512
STEP=DOWNSCALE//2
THR=0.75

# Loop to compute the metrics on evaluation
tests=os.listdir(test_path)
Preds={}
for t in tests:
    wells=os.listdir(os.path.join(test_path,t))
    for w in wells:
        if os.path.exists(os.path.join(test_path,t,w,"Mask_1.png")):
            img=os.path.join(test_path,t,w,"INorm_rgb.png") #eg: /work3/s174182/Test_data\\burkholderiaaccuracy20220927090618\\A1_D4_1\\INorm.png
            mask=os.path.join(test_path,t,w,"Mask_1.png")
            Preds[img]=evaluate(img,mask,model,threshold=THR,DEVICE=DEVICE,step=STEP, downscale=DOWNSCALE)

# initialize and load into variables
dice=0; acc=0; prec=0; spec=0; rec=0
for m in list(Preds.keys()):
    dice+=Preds[m]["dice"]
    acc+=Preds[m]["Accuracy"]
    prec+=Preds[m]["Precision"]
    spec+=Preds[m]["Specificity"]
    rec+=Preds[m]["Recall"]

# Compute means
N=len(Preds)
dice=dice/N
acc=acc/N
prec=prec/N
spec=spec/N
rec=rec/N

# Save in json file
Preds["Average_metrics"]={"dice":dice,"Accuracy":acc,"Specificity":spec,"Precision":prec,"Recall":rec}
Preds["settings"]={"step":STEP,"Threshold":THR}
with open(f'../../reports/metrics{model_name}.json','w') as fp:
    json.dump(Preds,fp)