'''
This code dishonors all dignity that i have
'''

import numpy as np
from tqdm import tqdm
from make_dataset import BacteriaDataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import decimal
decimal.getcontext().prec=300
transform = transforms.ToTensor()
train_dir="/work3/s174182/train_data/RGB_method_balanced_1/train/"

device      = torch.device('cpu') 
num_workers = 4
image_size  = 512 
batch_size  = 1
train_ds = BacteriaDataset(image_dir = train_dir, mask_dir=train_dir)
image_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)

# placeholders

N=2000
count = N*512**2

psum1=np.zeros(N)
psum_sq1=np.zeros(N)

psum2=np.zeros(N)
psum_sq2=np.zeros(N)

psum3=np.zeros(N)
psum_sq3=np.zeros(N)
# loop through images
j=0
for image,targets in tqdm(image_loader):
    inputs=image.float().numpy()

    img1=inputs[:,:,:,0]
    img2=inputs[:,:,:,1]
    img3=inputs[:,:,:,2]

    psum1[j]    = img1.sum()
    psum_sq1[j] = decimal.Decimal(str((img1**2).sum()))

    psum2[j]    = img2.sum()
    psum_sq2[j] = decimal.Decimal(str((img2**2).sum()))

    psum3[j]    = img3.sum()
    psum_sq3[j] = decimal.Decimal(str((img3**2).sum()))
    if j==N-1:
        break
    j+=1





# mean and std
total_mean1 = (psum1 / count).sum()
total_var1  = (psum_sq1 / count).sum() - (total_mean1 ** 2)
total_std1  = np.sqrt(total_var1)

total_mean2 = (psum2 / count).sum()
total_var2  = (psum_sq2 / count).sum() - (total_mean2 ** 2)
total_std2  = np.sqrt(total_var2)

total_mean3 = (psum3 / count).sum()
total_var3  = (psum_sq3 / count).sum() - (total_mean3 ** 2)
total_std3  = np.sqrt(total_var3)

# output

print('mean1: '  + str(total_mean1))
print('std1:  '  + str(total_std1))

print('mean2: '  + str(total_mean2))
print('std2:  '  + str(total_std2))

print('mean3: '  + str(total_mean3))
print('std3:  '  + str(total_std3))
