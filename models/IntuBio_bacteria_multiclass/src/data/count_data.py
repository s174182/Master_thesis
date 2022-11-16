import os
import numpy as np
import cv2 


# Main image directory
main_directory="/work3/s174182/multiclass_data_patch/multiclass/train/"
save_prob = 0.1
# Get in the main directory and go through the samples  
folders=os.listdir(main_directory)
# counter
count_blacks = 0
count_all = 0
count_blacks_balanced = 0
for f in folders:
    # Go through wells A-C
    sub_folders=os.listdir(os.path.join(main_directory,f))
    for sf in sub_folders:
        # go through all normalized images
        # for imgs in os.listdir(os.path.join(main_directory, f, sf, "img/")):
        #     break
        #     print(imgs)

        
        for masks in os.listdir(os.path.join(main_directory, f, sf, "mask/")):
            # count number of masks with no bacteria in them
            count_all += 1
            mask = cv2.imread(os.path.join(main_directory, f, sf, "mask/", masks), 0)

            if mask.sum() < 1:
                count_blacks += 1
                


print(f"{count_blacks} images with no bacteria")
print(f"{count_all} images in total" )
print(f"{count_blacks/count_all * 100:.2f}% images with no bacteria")
