# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:42:22 2022

@author: Jonat
"""

import os
import shutil
from sklearn.model_selection import train_test_split

input_directory="/work3/s174182/No_padding/"

traindata_path="/work3/s174182/train_data/"
#print(os.path.join(img_orig,"train"))
# os.makedirs(os.path.join(traindata_path,"train"), exist_ok=True) # create folders for later use
# os.makedirs(os.path.join(traindata_path,"val"), exist_ok=True)
# os.makedirs(os.path.join(traindata_path,"test"), exist_ok=True)

folders=os.listdir(input_directory)
totalimg=[]
totalmask=[]


for f in folders:
    sub_folders=os.listdir(os.path.join(input_directory,f))
    for sf in sub_folders:

        img_folder=os.listdir(os.path.join(input_directory,f,sf,'img'))
        mask_folder=os.listdir(os.path.join(input_directory,f,sf,'mask'))
        
        img_orig =os.path.join(input_directory,f,sf,'img')
        mask_orig =os.path.join(input_directory,f,sf,'mask')

        new_mask_list = [mask_orig +"/"+ x for x in mask_folder]
        new_img_list = [img_orig+"/"+x for x in img_folder]
        totalimg.append(new_img_list)
        totalmask.append(new_mask_list)
        
y=sorted(sum(totalmask,[]))
X=sorted(sum(totalimg,[]))


X_train, X_val, y_train, y_val= train_test_split(X, y, test_size=0.2, random_state=1)


'''

Rename training and test data to appropriate folders.

'''

# Training
output_directory="/work3/s174182/train_data/no_padding/train/"
for i in range(len(X_train)):
    maskname=y_train[i]
    imgname=X_train[i]
    
    destmask=output_directory+"/".join(maskname.split("/")[-4:])
    destimg=output_directory+"/".join(imgname.split("/")[-4:])


    os.makedirs("/"+"/".join(destmask.split("/")[1:-1]), exist_ok=True)
    os.makedirs("/"+"/".join(destimg.split("/")[1:-1]), exist_ok=True)

    shutil.copy(str(maskname), destmask)
    shutil.copy(str(imgname),destimg)
    

    
#Val
output_directory="/work3/s174182/train_data/no_padding/val/"
for i in range(len(X_val)):
    maskname=y_val[i]
    imgname=X_val[i]
    destmask=output_directory+"/".join(maskname.split("/")[-4:])
    destimg=output_directory+"/".join(imgname.split("/")[-4:])

    os.makedirs("/"+"/".join(destmask.split("/")[1:-1]), exist_ok=True)
    os.makedirs("/"+"/".join(destimg.split("/")[1:-1]), exist_ok=True)

    shutil.copy(str(maskname), destmask)
    shutil.copy(str(imgname),destimg)
