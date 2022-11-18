import os
import numpy as np
import cv2 


# Main image directory
main_directory="/work3/s174182/multiclass_data/multiclass/"
from sklearn.utils import class_weight

# Get in the main directory and go through the samples
folders=os.listdir(main_directory)
# counter

count_black_pixels = 0
count_white_pixels = 0
count = 0
mask_reshaped = []
for f in folders:
    # Go through wells A-C
    if '.DS_Store' in f:
        continue 
    sub_folders=os.listdir(os.path.join(main_directory,f))
    for sf in sub_folders:
        if os.path.exists(os.path.join(main_directory, f, sf, "Mask2.png")):
            mask = cv2.imread(os.path.join(main_directory, f, sf, "Mask2.png"), 0)
            mask_reshaped.append(mask.reshape(-1,1))
            
            count += 1
        
        if count > 10:
            break
    if count > 10:
        break

mask_reshaped = np.array(mask_reshaped)
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(mask_reshaped.flatten()), y=mask_reshaped.flatten())
print(f"background weight: {class_weights[0]}\n CFU weight: {class_weights[1]}\n Bubble weight: {class_weights[2]}\n Displacement weight: {class_weights[3]}\n Others weight: {class_weights[4]}")