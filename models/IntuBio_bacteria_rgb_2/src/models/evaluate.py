# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:32:58 2022

@author: Jonat

This script is meant for evalutation, it loads a full Image together with its mask.
Then it pads the image to preserve the full image, crop it into patches,
predicts on those sticth them back together and measure the dice score.
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

def evaluate(img_path,mask_path,model,DEVICE='cpu', threshold=0.5, step=256):
    print(img_path)
    transform = transforms.ToTensor()
    #model.eval()

    DOWNSCALE=512 # this is equal to the patch size
    
    # Read image and convert from BGR to RGB
    img_orig=cv2.imread(img_path, cv2.COLOR_RGB2BGR)
    N,M,_=np.shape(img_orig)

    # Mask
    mask = cv2.imread(mask_path, 0)
    mask[mask>1]=1
    
    #Pad the image, nessecary in order to crop the image into 512X512 patches
    hor_pad=np.ceil(img_orig.shape[1]/DOWNSCALE)*DOWNSCALE #padding in horizontal direction
    ver_pad=np.ceil(img_orig.shape[0]/DOWNSCALE)*DOWNSCALE #padding in vertical direction
    img=np.zeros((int(ver_pad),int(hor_pad),3))
    img[0:img_orig.shape[0],0:img_orig.shape[1],:]=img_orig
    
    # Normalize image as to match training
    pred_transform = A.Compose([A.augmentations.transforms.Normalize(mean=(144.8, 147.22, 149.29), std=(46.7, 45.58, 44.91), max_pixel_value=1, always_apply=False, p=1.0),
    ])
    augmented = pred_transform(image=img)
    img=augmented["image"]

    # Create the patches
    patches=patchify(img,(DOWNSCALE,DOWNSCALE,3),step=step)

    # Loop through and predict each patch
    pred_patches=[]
    with torch.no_grad():
        for i in range(patches.shape[0]):
            for j in range (patches.shape[1]):
                x = transform(patches[i, j, 0, :, :, :])
                x = x[None,:].float().to(device=DEVICE) #as its not a batch do a dummy expansion
                preds = model(x)
                preds = torch.sigmoid(preds)
                pred_patches.append(preds)

    pred_patches = np.array([pred_patches[k].cpu().numpy().squeeze() for k in range(0,len(pred_patches))])
    pred_mask =recon_im(pred_patches,img.shape[0],img.shape[1],1,step)
    pred_mask = (pred_mask[:img_orig.shape[0],:img_orig.shape[1]])
    heatmap_copy=pred_mask
    pred_mask= (pred_mask>threshold).astype(np.uint8)
    
    #Metrics
    TP = np.sum(np.logical_and(pred_mask == 1, mask == 1))
    TN = np.sum(np.logical_and(pred_mask == 0, mask == 0))
    FN = np.sum(np.logical_and(pred_mask == 0, mask == 1))   
    FP = np.sum(np.logical_and(pred_mask == 1, mask == 0))

    dice = 2 * (TP)/(FP+2*TP+FN+1e-8)
    ACC = (TP+TN)/(TP+TN+FP+FN)
    Spec = (TN/(FP+TN+1e-8))
    Prec = TP/(TP+FP+1e-8)
    Recall= TP/(TP+FN+1e-8)
    
    Metrics={"dice":dice,"Accuracy":ACC,"Specificity":Spec,"Precision":Prec,"Recall":Recall}

    #### SAVE IMAGE ####
    save_folder="_".join(img_path.split("/")[-3:-1])
    shapes=np.zeros((N,M,3)).astype(np.uint8)
    shapes[pred_mask==1,2]=255 # create a filter for the predicted masks
    predicted_img = cv2.addWeighted(img_orig,1,shapes,0.25,0.1)
   
    os.makedirs("/work3/s174182/predictions/"+model_name,exist_ok=True)
   
    cv2.imwrite("/work3/s174182/predictions/"+model_name+"/"+save_folder+".png",predicted_img)
    plt.ioff()
    svm = sn.heatmap(heatmap_copy,yticklabels=False,xticklabels=False)
    figure = svm.get_figure()    
    figure.savefig("/work3/s174182/predictions/"+model_name+"/"+save_folder+"prob_map"+".png", dpi=1555)
    plt.close()

    return Metrics

test_path="/work3/s174182/Test_data/"
model_name="graceful-mountain-41"
model_path="../../models/"+model_name+".pth" #to be made as an argument
os.makedirs("../../data/predictions/"+model_name,exist_ok=True) # to save predicted images in

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(in_channels=3, out_channels=1)

checkpoint=torch.load(model_path)  
model.to(device=DEVICE)
load_checkpoint(checkpoint,model) 


STEP=512//2
THR=0.75
tests=os.listdir(test_path)
Preds={}
for t in tests:
    wells=os.listdir(os.path.join(test_path,t))
    for w in wells:
        if os.path.exists(os.path.join(test_path,t,w,"Mask_1.png")):
            img=os.path.join(test_path,t,w,"INorm_rgb.png") #eg: /work3/s174182/Test_data\\burkholderiaaccuracy20220927090618\\A1_D4_1\\INorm.png
            mask=os.path.join(test_path,t,w,"Mask_1.png")
            Preds[img]=evaluate(img,mask,model,threshold=THR,DEVICE=DEVICE,step=STEP)


dice=0; acc=0; prec=0; spec=0; rec=0

for m in list(Preds.keys()):
    dice+=Preds[m]["dice"]
    acc+=Preds[m]["Accuracy"]
    prec+=Preds[m]["Precision"]
    spec+=Preds[m]["Specificity"]
    rec+=Preds[m]["Recall"]

N=len(Preds)
dice=dice/N
acc=acc/N
prec=prec/N
spec=spec/N
rec=rec/N

Preds["Average_metrics"]={"dice":dice,"Accuracy":acc,"Specificity":spec,"Precision":prec,"Recall":rec}
Preds["settings"]={"step":STEP,"Threshold":THR}
with open(f'../../reports/metrics{model_name}.json','w') as fp:
    json.dump(Preds,fp)