import torch
import cv2
from model import Unet
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import time 
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
from scipy import ndimage


# Define a transform to convert the image to tensor
transform = transforms.ToTensor()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DOWNSCALE=512 #size do be patched into

##### MODEL ######
def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
model = Unet()
model_path="../../models/2022-10-27 10_27_58.506022.pth"#to be made as an argument
checkpoint=torch.load(model_path)  
model.to(device=DEVICE)
load_checkpoint(checkpoint,model)

###### LOAD img #########
img_ref="D:/NIIMBL_TEST2/vegetativemixAccuracy20220930062912/Analysis/B4_D2_2/TotalProjI_R2.png"
img_path="C:/Users/Jonat/OneDrive/Documents/DTU fag/Master_speciale/IntuBio_dataset/Test_masks/pseudomonasprecision dilutions 3&420221013094047/A1_D4_1/INorm.png"
img_ref=cv2.imread(img_ref,0)
img_orig=cv2.imread(img_path,0)

#img_orig=normimg(img_orig,img_ref)


hor_pad=np.ceil(img_orig.shape[1]/DOWNSCALE)*DOWNSCALE #padding in horizontal direction
ver_pad=np.ceil(img_orig.shape[0]/DOWNSCALE)*DOWNSCALE #padding in vertical direction
img=np.zeros((int(ver_pad),int(hor_pad)))
img[0:img_orig.shape[0],0:img_orig.shape[1]]=img_orig




patches=patchify(img,(DOWNSCALE,DOWNSCALE),step=DOWNSCALE) 
# Convert the image to PyTorch tensor
pred_patches=[]
startTime = time.time()
with torch.no_grad():
    for i in range(patches.shape[0]):
        for j in range (patches.shape[1]):
            print(i,j)
            x = transform(patches[i,j,:,:])
            x = x[None,:].float().to(device=DEVICE) #as its not a batch do a dummy expansion
            preds = model(x)
            preds = (preds>0.5).float()
            pred_patches.append(preds)




pred_patches = np.array([pred_patches[k].cpu().numpy().squeeze() for k in range(0,len(pred_patches))])
pred_patches_reshaped = np.reshape(pred_patches, (patches.shape[0], patches.shape[1], DOWNSCALE,DOWNSCALE) )
reconstructed_image = unpatchify(pred_patches_reshaped, img.shape)
reconstructed_image = reconstructed_image[:img_orig.shape[0],:img_orig.shape[1]]

reconstructed_image = ROI(reconstructed_image.astype(np.uint8),img_orig)
endTime = time.time()
print("Total time",endTime-startTime)

fig, axs=plt.subplots(1,2)
axs[0].imshow(img_orig,cmap="gray")
axs[1].imshow(reconstructed_image,cmap="gray")
plt.show()
