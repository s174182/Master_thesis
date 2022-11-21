import cv2
import numpy as np
from scipy import ndimage

 """
    Helper functions
"""

def ROI(mask,img):
    IthrMask=18
    th, im_th = cv2.threshold(img[:,:,2], IthrMask, 255, cv2.THRESH_BINARY)
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