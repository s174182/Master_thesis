# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:42:22 2022

@author: Jonat
"""

import os
from sklearn.model_selection import train_test_split

input_directory="D:\Annotated_segmentation_patch"
output_directory="D:\train_data"

folders=os.listdir(input_directory)
totalimg=[]
totalmask=[]
for f in folders:
    print(f)
    sub_folders=os.listdir(os.path.join(input_directory,f))
    for sf in sub_folders:
        img_folder=os.listdir(os.path.join(input_directory,f,sf,'img'))
        mask_folder=os.listdir(os.path.join(input_directory,f,sf,'mask'))
        
        mask_orig =os.path.join(input_directory,f,sf,'mask')
        img_orig =os.path.join(input_directory,f,sf,'mask')
        
        new_mask_list = [mask_orig +"/"+ x for x in mask_folder]
        new_img_list = [img_orig+"/"+x for x in img_folder]
        totalimg.append(new_mask_list)
        totalmask.append(new_mask_list)
        
X=sum(totalmask,[])
y=sum(totalimg,[])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.2, random_state=1)
        



'''

Rename training and test data to appropriate folders.

'''

# Training
for i in len(X_train):
    maskname=X_train[i]
    imgname=y_train[i]

    os.makedirs(os.path.dirname(), exist_ok=True)
    shutil.copy(src_fpath, dest_fpath)
    
    
    