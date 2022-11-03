import os
import numpy as np
import cv2 

# Main image directory
main_directory="/work3/s174182/train_data/RGB_method_balanced/val/"

# mean values list
mean_vals = []
std_vals = []

# Get in the main directory and go through the samples
folders=os.listdir(main_directory)
for f in folders:
    # Go through wells A-C
    sub_folders=os.listdir(os.path.join(main_directory,f))
    for sf in sub_folders:
        # go through all normalized images
        for imgs in os.listdir(os.path.join(main_directory, f, sf, "img/")):
            # read image and compute standard deviation and mean
            cv2.imread(os.path.join(main_directory,f,sf,"img",imgs))
            mean_vals.append(np.mean(img))
            std_vals.append(np.std(img))
        


print(f"Global mean of images: {np.mean(mean_vals)}")
print(f"Global standard deviation of images: {np.mean(std_vals)}")
