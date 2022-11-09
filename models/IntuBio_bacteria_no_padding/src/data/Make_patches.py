''' 


'''

from patchify import patchify
import cv2
import os
import numpy as np
from PIL import Image
from helpers import normimg

N=572
step=388

main_directory="/work3/s174182/Orig_dataset/Orig_data_filtered/"
Save_directory="/work3/s174182/No_padding"

folders=os.listdir(main_directory)

for f in folders:
    sub_folders=os.listdir(os.path.join(main_directory,f))
    for sf in sub_folders:
        if os.path.exists(os.path.join(main_directory,f,sf,"mask_1.png")):
            L=len(os.listdir(os.path.join(main_directory,f,sf)))-1
            mask=np.array(cv2.imread(os.path.join(main_directory,f,sf,"mask_1.png"),0))
            im2=np.array(cv2.imread(os.path.join(main_directory,f,sf,"TotalProjI_R2.png"),0))
            imT=np.array(cv2.imread(os.path.join(main_directory,f,sf,f"TotalProjI_R{L}.png"),0))
            norm_img=normimg(imT,im2)
            n,m=mask.shape[0],mask.shape[1]
            
            # crop image such that 388 stepsize can be applied with size 572x572 of the images but 388X388 masks

            mask_pad=np.zeros((5432,6208))
            mask_pad[56:5319+56,44:6119+44]=mask

            img_pad=np.zeros((5616,6392))
            img_pad[148:5319+148:,136:(6119+136)]=norm_img
            
            mask_patches=patchify(mask_pad,(388,388),step=step)
            norm_patches=patchify(img_pad,(572,572),step=step)
            
            save_mask_folder=os.path.join(Save_directory,f,sf,"mask")
            save_img_folder=os.path.join(Save_directory,f,sf,"img")
            os.makedirs(save_img_folder,exist_ok=True)
            os.makedirs(save_mask_folder,exist_ok=True)
            for i in range(mask_patches.shape[0]):
                for j in range(mask_patches.shape[1]):
                    num = i * mask_patches.shape[1] + j
                    
                    #mask
                    patchm = mask_patches[i, j, :,:]
                    patchm = Image.fromarray(patchm.astype(np.uint8))
                    #image
                    patchI = norm_patches[i, j, :,:]
                    patchI = Image.fromarray(patchI.astype(np.uint8))
                   
                    #Save

                    patchm.save(save_mask_folder+"/"+f"MaskP_{num}.png")
                    patchI.save(save_img_folder+"/"+f"NormIP_{num}.png")

                
            
        
