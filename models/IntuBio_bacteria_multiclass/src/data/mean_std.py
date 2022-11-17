import numpy as np
from tqdm import tqdm
from make_dataset import BacteriaDataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import decimal
decimal.getcontext().prec=300
transform = transforms.ToTensor()
train_dir="/work3/s174182/train_data/Annotated_segmentation_patch_balanced/train/"

device      = torch.device('cpu') 
num_workers = 4
image_size  = 512 
batch_size  = 1
train_ds = BacteriaDataset(image_dir = train_dir, mask_dir=train_dir)
image_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)

# placeholders

N=2000
count = N*512**2

psumo=np.zeros(N)
psum_sqo=np.zeros(N)
# loop through images
j=0
for image,targets in tqdm(image_loader):
    inputs=image.float().numpy()


    psumo[j]    = inputs.sum()
    psum_sqo[j] = decimal.Decimal(str((inputs**2).sum()))
    if j==N-1:
        break
    j+=1



# print(psume)
# print(psum_sqe)    
   
# pixel count


# mean and std
total_mean = (psumo / count).sum()
total_var  = (psum_sqo / count).sum() - (total_mean ** 2)
total_std  = np.sqrt(total_var)

# output
print("total variation " + str(total_var))
print('mean: '  + str(total_mean))
print('std:  '  + str(total_std))
