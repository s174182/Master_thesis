import cv2
import numpy as np
from scipy import ndimage
import os
'''
As we experienced a lot of single pi
'''
def cc(mask,filename):
    count=0
    analysis=cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)
    (totalLabels,label_ids,values,centroid)=analysis    
    for i in range(1,totalLabels):
        area=values[i, cv2.CC_STAT_AREA]
        
        if area<25:
            count+=1

    return count

Overallcount=0
path="/work3/s174182/train_data/no_padding/train/"
tests=os.listdir(path)

for t in tests:
    wells=os.listdir(os.path.join(path,t))
    for w in wells:
        masks=os.listdir(os.path.join(path,t,w,"mask"))
        for m in masks:
            file=cv2.imread(os.path.join(path,t,w,"mask",m),0)
            Overallcount+=cc(file,os.path.join(path,t,w,"mask",m))

print(Overallcount)