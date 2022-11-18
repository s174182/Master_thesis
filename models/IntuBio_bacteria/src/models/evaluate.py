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
sys.path.append('../data/')
from helpers import ROI
from utils import (load_checkpoint, get_patches, recon_im)
import seaborn as sn



def evaluate(img_path,mask_path,model,threshold,DEVICE='cpu',step=256):
    print(img_path)
    transform = transforms.ToTensor()
    
    DOWNSCALE=512 # this is equal to the patch size
    
    # Read image and convert from BGR to RGB
    img_orig=cv2.imread(img_path,0)
    N,M=np.shape(img_orig)
    mask=cv2.imread(mask_path,0)
    mask[mask>1]=1
    
    #Pad the image, nessecary in order to crop the image into 512X512 patches
    hor_pad=np.ceil(img_orig.shape[1]/DOWNSCALE)*DOWNSCALE #padding in horizontal direction
    ver_pad=np.ceil(img_orig.shape[0]/DOWNSCALE)*DOWNSCALE #padding in vertical direction
    img=np.zeros((int(ver_pad),int(hor_pad)))
    img[0:img_orig.shape[0],0:img_orig.shape[1]]=img_orig
    
    # Create the patches
    patches=patchify(img,(DOWNSCALE,DOWNSCALE),step=step)
    
    # Loop through and predict each patch
    pred_patches=[]
    with torch.no_grad():
        for i in range(patches.shape[0]):
            for j in range (patches.shape[1]):
                x = transform(patches[i, j,:,:])
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
    img_orig=cv2.merge((img_orig,img_orig,img_orig))
    shapes=np.zeros((N,M,3)).astype(np.uint8)
    shapes[pred_mask==1,0]=255;shapes[pred_mask==1,1]=255 # create a bluish filter for the predicted masks
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
model_name= sys.argv[1]#"2022-10-27 10_24_52.652741"
model_path="../../models/"+model_name+".pth" #to be made as an argument

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(in_channels=1)

checkpoint=torch.load(model_path)  
model.to(device=DEVICE)
load_checkpoint(checkpoint,model) 

STEP=388//2
THR=0.75
tests=os.listdir(test_path)
Preds={}
for t in tests:
    wells=os.listdir(os.path.join(test_path,t))
    for w in wells:
        if os.path.exists(os.path.join(test_path,t,w,"Mask_1.png")):
            img=os.path.join(test_path,t,w,"INorm.png") #eg: /work3/s174182/Test_data\\burkholderiaaccuracy20220927090618\\A1_D4_1\\INorm.png
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