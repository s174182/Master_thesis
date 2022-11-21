''' 
This script is used to create patches of NxN for the purpose of using the patches as training data

Original images have the dimensions of

6119 X 5319

Args:
    N: Dimension for NxN image patch
    step: Stepsize between patches

'''

from patchify import patchify
import cv2
import os
import numpy as np
from PIL import Image

N=512
step=256
main_directory="/work3/s174182/multiclass_data/multiclass/"
Save_directory="/work3/s174182/multiclass_data_patch/multiclass/"

folders=os.listdir(main_directory)

print("Creating patches...")
for f in folders:
    if '.DS_Store' in f:
        continue
    sub_folders=os.listdir(os.path.join(main_directory,f))
    for sf in sub_folders:
        if os.path.exists(os.path.join(main_directory,f,sf,"Mask2.png")):
            mask=np.array(cv2.imread(os.path.join(main_directory,f,sf,"Mask2.png"),0))
            norm_img=np.array(cv2.imread(os.path.join(main_directory,f,sf,"INorm.png"),0))
            n,m=mask.shape[0],mask.shape[1]
            
            # crop image such that 256 stepsize can be applied
            hcrop=((m-m//N*N)//2)
            redmask=mask[0:n//N*N,hcrop+1:m-hcrop]
            redimg=norm_img[0:n//N*N,hcrop+1:m-hcrop]
            
            mask_patches=patchify(redmask,(N,N),step=step)
            norm_patches=patchify(redimg,(N,N),step=step)
            
            save_mask_folder=os.path.join(Save_directory,f,sf,"mask")
            save_img_folder=os.path.join(Save_directory,f,sf,"img")
            os.makedirs(save_img_folder)
            os.makedirs(save_mask_folder)
            for i in range(mask_patches.shape[0]):
                for j in range(mask_patches.shape[1]):
                    num = i * mask_patches.shape[1] + j
                    
                    #mask
                    patchm = mask_patches[i, j, :,:]
                    patchm = Image.fromarray(patchm)
                    #image
                    patchI = norm_patches[i, j, :,:]
                    patchI = Image.fromarray(patchI)
                   
                    #Save

                    patchm.save(save_mask_folder+"/"+f"MaskP_{num}.png")
                    patchI.save(save_img_folder+"/"+f"NormIP_{num}.png")
                
            
        



"""


import cv2
import numpy
import matplotlib.pyplot as plt
import numpy as np


pathM="D:\Annotated_segmentation\PA_accuracy_linearity_20220603-11140806\B2_D3_\mask_1.png"
pathI="D:\Annotated_segmentation\PA_accuracy_linearity_20220603-11140806\B2_D3_\Inorm.png"


stepsize=256


mask=np.array(cv2.imread(pathM,0))
img=np.array(cv2.imread(pathI,0))

n=mask.shape[0]
m=mask.shape[1]

#Make expansion
eximg=np.zeros((5632,6144))
exmask=np.zeros((5632,6144))
eximg[0:n,0:m]=img
exmask[0:n,0:m]=mask

#Make reduction
redmask=mask[0:5120,244:m-243]
redimg=img[0:5120,244:m-243]


print(redimg.shape)


fig, axs=plt.subplots(2,3)
axs[0,0].imshow(img)
axs[0,1].imshow(eximg)
axs[1,0].imshow(mask)
axs[1,1].imshow(exmask)
axs[1,2].imshow(redmask)
axs[0,2].imshow(redimg)


"""