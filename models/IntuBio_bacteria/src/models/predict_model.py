import torch
import cv2
from model import Unet
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import time 
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify

# Define a transform to convert the image to tensor
transform = transforms.ToTensor()
DEVICE = "cpu"#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DOWNSCALE=2048 #size do be patched into
N=115 #blur factor
##### MODEL ######
def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
model = Unet()
model_path="2022-10-14 14:27:16.324975.pth"#to be made as an argument
checkpoint=torch.load(model_path)  
model.to(device=DEVICE)
load_checkpoint(checkpoint,model)

###### LOAD img #########
img_path="TotalProjI_R14.png"
img_orig=cv2.imread(img_path,0)
img=cv2.resize(img_orig,((img_orig.shape[0]//DOWNSCALE)*DOWNSCALE,(img_orig.shape[1]//DOWNSCALE)*DOWNSCALE))


img=cv2.GaussianBlur(img,(N,N),0)


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
            pred = model(x)
            preds = torch.sigmoid(pred)
            preds = (preds>0.9995).float()
            pred_patches.append(preds)

    img_scale=cv2.resize(img_orig,(1024,1024))
    x = transform(img_scale)
    x = x[None,:].float().to(device=DEVICE) #as its not a batch do a dummy expansion
    scaled_pred = model(x)
    scaled_pred = torch.sigmoid(scaled_pred)
    scaled_pred = (scaled_pred>0.9995).float().cpu().numpy().squeeze() 
#pred_patches = np.array(pred_patches)


pred_patches = np.array([pred_patches[k].cpu().numpy().squeeze() for k in range(0,len(pred_patches))])
print(pred_patches[0].shape)
pred_patches_reshaped = np.reshape(pred_patches, (patches.shape[0], patches.shape[1], DOWNSCALE,DOWNSCALE) )
reconstructed_image = unpatchify(pred_patches_reshaped, img.shape)
endTime = time.time()
print("Total time",endTime-startTime)

plt.figure(figsize=(18,18))
plt.subplot(221)
plt.title('Large Image')
plt.imshow(img, cmap='gray')
plt.subplot(222)
plt.title('Prediction of large Image')
plt.imshow(reconstructed_image, cmap='gray')
plt.subplot(223)
plt.title("Rescaled image")
plt.imshow(img_scale,cmap="gray")
plt.subplot(224)
plt.title("prediction of rescaled image")
plt.imshow(scaled_pred,cmap="gray")
plt.show()