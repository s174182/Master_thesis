from tkinter import image_names
import numpy as np
import shutil
import os
import cv2


Intubio_bacteria_path="/work3/s174182/train_data/Annotated_segmentation_patch_balanced"
IntuBio_bacteria_rgb="/work3/s174182/RGB_method"

IntuBio_train_folder="/work3/s174182/train_data/RGB_method_balanced_1/train/"

Train=Intubio_bacteria_path+"/train/"
Val=Intubio_bacteria_path+"/val/"


train_folders=os.listdir(Train)

for f in train_folders: #loop igennem forskellige test eg. staph LOQ4
    tests=os.listdir(os.path.join(Train,f)) #list alle brønde i annotated
    tests_rgb=os.listdir(os.path.join(IntuBio_bacteria_rgb,f)) #list alle brønde i Rgb
    test_folder=zip(tests,tests_rgb)
    for sf,sf_rgb in test_folder:
        well=os.path.join(Train,f,sf) #path til en brønd eg: /work3/s174182/train_data/Annotated_segmentation_patch_balanced/train/BC_accuracy_linearity_20220607-14230983/A1_D4_/
        print(well)
        images=sorted(os.listdir(os.path.join(well,"img"))) # alle billeder i brønden
        masks=sorted(os.listdir(os.path.join(well,"mask"))) # alle masker i brøndene

        rgb_path=os.path.join(IntuBio_bacteria_rgb,f,sf_rgb) # path til rgb billeder /work3/s174182/RGB_method/BC_accuracy_linearity_20220607-14230983/A1_D4_1/

        #Create a folder where to move the new RGB images
        rgb_imgdest=os.path.join(IntuBio_train_folder,f,sf,"img")
        #os.makedirs(rgb_imgdest,exist_ok=True)
        rgb_maskdest=os.path.join(IntuBio_train_folder,f,sf,"mask")
        #os.makedirs(rgb_maskdest,exist_ok=True)
        
        im=zip(images,masks)
        for i,m in im:
            rgb_maskname=os.path.join(rgb_path,"mask",m)
            rgb_imgname=os.path.join(rgb_path,"img",i)
            print(i,m)
            destmask=os.path.join(rgb_maskdest,m)
            print(rgb_maskname,destmask)
            destimg=os.path.join(rgb_imgdest,i)
            #shutil.copy(str(rgb_maskname), destmask)
            #shutil.copy(str(rgb_imgname),destimg)
