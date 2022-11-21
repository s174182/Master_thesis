import cv2
import numpy as np
from scipy import ndimage

"""
Neat helper functions to beautify training cycle

ROI: Computes region of interest, i.e., everything else than plate is black
normimg: Normalizes image using rep 2 as reference image 
"""

def ROI(mask,img):
    IthrMask=18
    th, im_th = cv2.threshold(img, IthrMask, 255, cv2.THRESH_BINARY);
    im_th=ndimage.binary_fill_holes(im_th,structure =None,output =None,origin=0).astype(np.uint8)
    
    kernel = np.ones((25, 25), np.uint8)
      
    # # Using cv2.erode() method 
    ROI = cv2.erode(im_th, kernel)

    mask[ROI==0]=0
    analysis=cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)
    (totalLabels,label_ids,values,centroid)=analysis
    output=np.zeros(mask.shape,dtype="uint8")
    
    for i in range(1,totalLabels):
        area=values[i, cv2.CC_STAT_AREA]
        
        if area>60:
            componentMask = (label_ids==i).astype("uint8")
            output=cv2.bitwise_or(output,componentMask)
            
    return output


def normimg(img,imgref):
    IthrMask=18
    imnorm=(170*(img-10).astype(float)/(imgref-10).astype(float))
    imnorm[imnorm>255]=255
    imnorm=imnorm.astype(np.uint8)

    
    th, im_th = cv2.threshold(imgref, IthrMask, 255, cv2.THRESH_BINARY);
    im_th=ndimage.binary_fill_holes(im_th,structure =None,output =None,origin=0).astype(np.uint8)
    
    kernel = np.ones((25, 25), np.uint8)
      
    # # Using cv2.erode() method 
    ROI = cv2.erode(im_th, kernel)

    imnorm[ROI==0]=0
    return imnorm 
