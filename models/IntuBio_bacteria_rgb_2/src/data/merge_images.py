# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 10:52:34 2022

@author: Jonat

Merges images needed for the RGB method
Takes three timepoints and stacks it as T, T-2, T-4 for an RGB model interpretation

"""

from patchify import patchify
import pdb;
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
main_directory="/work3/s174182/Orig_dataset/Orig_data_filtered/"

Save_directory="/work3/s174182/RGB_method_nopad/"

folders=os.listdir(main_directory)

def normimg(img,imgref):
    IthrMask=18
    imnorm=(170*(img-10).astype(float)/(imgref-10).astype(float))
    imnorm[imnorm>255]=255
    imnorm=imnorm.astype(np.uint8)

    
    th, im_th = cv2.threshold(imgref, IthrMask, 255, cv2.THRESH_BINARY);
    im_th=ndimage.binary_fill_holes(im_th,structure =None,output =None,origin=0).astype(np.uint8)
    
    kernel = np.ones((25, 25), np.uint8)
      
    # # Using cv2.erode() method 
    ROI = cv2.erode(im_th, kernel)

    imnorm[ROI==0]=0
    return imnorm        


def patches(norm_img,path,Save_directory,N=572,step=388):
    mask=np.array(cv2.imread(os.path.join(path,"mask_1.png"),0))
    n,m=mask.shape[0],mask.shape[1]
    
    # crop image such that 256 stepsize can be applied
    hcrop=((m-m//N*N)//2)
    redmask=mask[0:n//N*N,hcrop+1:m-hcrop]
    redimg=norm_img[0:n//N*N,hcrop+1:m-hcrop,:]
    
    mask_patches=patchify(redmask,(N,N),step=step)
    norm_patches=patchify(redimg,(N,N,3),step=step)

    
    save_mask_folder=os.path.join(Save_directory,f,sf,"mask")
    save_img_folder=os.path.join(Save_directory,f,sf,"img")
    os.makedirs(save_img_folder,exist_ok=True)
    os.makedirs(save_mask_folder,exist_ok=True)
    for i in range(mask_patches.shape[0]):
        for j in range(mask_patches.shape[1]):
            num = i * mask_patches.shape[1] + j
            
            #mask
            patchm = mask_patches[i, j, :,:]
            patchm = Image.fromarray(patchm)
            #image
            patchI = norm_patches[i, j, 0,:,:,:]
            patchI = Image.fromarray(patchI)
           
            #Save

            patchm.save(save_mask_folder+"/"+f"MaskP_{num}.png")
            patchI.save(save_img_folder+"/"+f"NormIP_{num}.png")
      

for f in folders:
    sub_folders=os.listdir(os.path.join(main_directory,f))
    for sf in sub_folders:
        if os.path.exists(os.path.join(main_directory,f,sf,"mask_1.png")):
           
            well=os.path.join(main_directory,f,sf)
            images=os.listdir(well)
            L=len(images)-1 #we minus one because the mask is also in the folder
            
            im0_name="TotalProjI_R2.png"
            im0_path=os.path.join(well,im0_name)
            im0=cv2.imread(im0_path,0)
            
            im1_name="TotalProjI_R"+str(L)+".png" #image corresponding to the mask time
            im1_path=os.path.join(well,im1_name)
            im1o=cv2.imread(im1_path,0)
            im1norm=normimg(im1o,im0)
            
            
            im2_name="TotalProjI_R"+str(L-2)+".png" # 2 hours earlier
            im2_path=os.path.join(well,im2_name)
            im2o=cv2.imread(im2_path,0)
            im2norm=normimg(im2o,im0)
                 
            im3_name="TotalProjI_R"+str(L-4)+".png" # 4 hours earlier
            im3_path=os.path.join(well,im3_name)
            im3o=cv2.imread(im3_path,0)
            im3norm=normimg(im3o,im0)
            
            Concatimg=np.dstack((im1norm,im2norm,im3norm)) # stack the images along the 3rd dimension
            patches(Concatimg,well,Save_directory)
            

            
            

