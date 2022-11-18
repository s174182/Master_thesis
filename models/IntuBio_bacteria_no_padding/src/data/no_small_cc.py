'''
This script cleanses the dataset such that no patches will have small connected components.
Hopefully this attempt will remove lots of the small and failfully predicted pixels.
'''
import os
import cv2
import shutil
train_path="/work3/s174182/train_data/no_padding/train"
val_path  ="/work3/s174182/train_data/no_padding/val"

def no_small_components(path):
    tests=os.listdir(path)
    for t in tests:
        print(t)
        wells=os.listdir(os.path.join(path,t))
        for w in wells:
            masks=os.listdir(os.path.join(path,t,w,"mask"))
            for m in masks:
                move=1
                mask=cv2.imread(os.path.join(path,t,w,"mask",m),0)
                analysis=cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)
                (totalLabels,label_ids,values,centroid)=analysis
                for i in range(1,totalLabels):
                    area=values[i, cv2.CC_STAT_AREA]
                    if area<50:
                        move=0
                if move==1:
                    save_folder=path.split("/")
                    save_folder[-2]="no_padding_no_small_cc"
                    
                    save_folder=os.path.join(("/").join(save_folder),t,w)
                   
                    maskname_orig=os.path.join(path,t,w,"mask",m)
                    maskname_dest=os.path.join(save_folder,"mask",m)

                    imgname_orig=os.path.join(path,t,w,"img","NormIP"+m[5:])
                    
                    imgname_dest=os.path.join(save_folder,"img","NormIP"+m[5:])
                    
                    os.makedirs(os.path.join(save_folder,"img"),exist_ok=True)
                    os.makedirs(os.path.join(save_folder,"mask"),exist_ok=True)
                    shutil.copy(imgname_orig,imgname_dest)
                    shutil.copy(maskname_orig,maskname_dest)

#no_small_components(train_path)   
no_small_components(val_path)           