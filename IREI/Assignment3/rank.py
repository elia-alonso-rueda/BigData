#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 19:13:49 2022

@author: Miriam, Itziar and Elia

"""

import cv2
import glob as gb
import re
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import operator

#%% FUNCTIONS

def segmentation_threshold(img):
    img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # img_eq = cv2.equalizeHist(img_gray)
    def filter_image(image, mask):
        r = image[:,:,0] * mask
        g = image[:,:,1] * mask
        b = image[:,:,2] * mask
        return np.dstack([r,g,b])
    
    thresh = threshold_otsu(img_gray)
    img_otsu  = img_gray < thresh
    filtered = filter_image(img, img_otsu)
    return filtered

    
#Read and show images

# Define the variables
dict_images = {} # Create a dictionary to store the images
array_target = [] # Create a list to store the names of the images
pattern="db\.*.(\w+\d*).*" # Create a pattern to extract the names of the imahes


img_path = 'db/'
ext = ['png', 'jpg', 'jpeg'] # Add image formats here

# Create a list to store the paths of the images of the data base
files = []
[files.extend(gb.glob(img_path + '*.' + e)) for e in ext]

# Enter the path of the image that you want to compare
pathNewImage = input("Enter a new image: ")
# Read the image entered and resize it in a common size
newImage = cv2.resize(cv2.imread(pathNewImage),(299, 299))

# Read the images of the database and store its names
for path in files:
    # Use the pattern to extract the name of the images
    match = re.search(pattern, path)
    # Read the image and resize it in a common size
    img = cv2.resize(cv2.imread(path),(299, 299))
    dict_images[match.group(1)] = img


#%%

######################################## COLORS METHOD ########################################

print("\n")
print("1. Colors method \n")

# Normalize the the input image
newImage_norm = cv2.normalize(newImage, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
# Create the histogram for the input image
hist_newImage = cv2.calcHist([newImage_norm], [0], None, [20],[0, 256])

dict_hist = {}
# Create a loop to iterate the list of images
for img in dict_images:
    # Normalize the image
    img_norm = cv2.normalize(dict_images[img], None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
    # Create a histogram for every image and store it in a list
    hist = cv2.calcHist([img_norm], [0], None, [20],[0, 256])
    dict_hist[img] = hist

# Create a list to store the values of the comparations
dict_corr = {}
distances_table = [[0 for x in range(4)] for y in range(len(dict_hist))]
# Define a list with the names of the comparation methods
methods_names = ["Correlation", "Chi-square", "Intersection", "Bhattacharyya"]

# Create a loop to iterate the list of histograms
for hist in dict_hist:
    # Compare the histogram of the new image with the histograms of the images of the data base
    corr = cv2.compareHist(hist_newImage, dict_hist[hist], 0)
    # Store the result of the comparation
    dict_corr[hist] = round(corr, 4)

# Sort the images by absolute value of the correlation
corr_sorted = sorted(dict_corr, key=lambda dict_key: abs(dict_corr[dict_key]), reverse=True)
# Printing the list of the three images with higher correlation using loop
for w in corr_sorted:
    print(f"Image name: {w}, Correlaton: {dict_corr[w]}")

thresh = round(len(corr_sorted)*0.15)
names_thres = corr_sorted[0:thresh]
dict_thres = {}
print(f"Name of the {thresh} images with higher correlation: {names_thres}")

# Filter the dictionary of the images using the threshold
for img in dict_images:
    if (img in corr_sorted[0:thresh]):
        dict_thres[img] = dict_images[img]


#%%

######################################## GRADIENT METHOD ########################################

print("\n")
print("2. Gradient method \n")

# Calculate the gradient for the inut image
fd, hog_new_img = hog(newImage, orientations=9, pixels_per_cell=(6, 6), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
# Rescale the histogram for better display
hog_new_res = exposure.rescale_intensity(hog_new_img, in_range=(0, 10))


# Define the dictionary of hogs
dict_hog = {}

# Create a loop to iterate the list of images
for img in dict_thres:
    # Apply the method hog for each image
    fd, hog_img = hog(dict_thres[img], orientations=9, pixels_per_cell=(6, 6), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
    # Rescale histogram for better display 
    hog_res = exposure.rescale_intensity(hog_img, in_range=(0, 10))
    # Store the result in the dictionary
    dict_hog[img]=hog_res

# Define a dictionary to store the euclidean distances
distance_array={}
# Create a loop to iterate the list of hog images
for hog in dict_hog:
    # Calculate the euclidean distance
    distance = cv2.norm(hog_new_res, dict_hog[hog])
    # Store the names and the distances of the images in the dictionary
    distance_array[hog]=distance

# Sort the images by distance
distance_sorted = sorted(distance_array, key=distance_array.get, reverse=False)

# Printing the list of the three images with less distance using loop
for w in distance_sorted:
    print(f"Image name: {w}, Distance: {round(distance_array[w], 4)}")

print("Name of the three images with less distance: ", distance_sorted[0:3])


# Representate the input image and the Histogram of Oriented Gradients of the images with less distance

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8), sharex=True, sharey=True) 

# Representate the original input image
ax1.imshow(newImage, cmap=plt.cm.gray) 
ax1.set_title('Input image')

# Representate the Hogs of the images with less distances
ax2.imshow(dict_hog[distance_sorted[0]], cmap=plt.cm.gray) 
ax2.set_title(f"Histogram of Oriented Gradients: {distance_sorted[0]}")
ax3.imshow(dict_hog[distance_sorted[1]], cmap=plt.cm.gray) 
ax3.set_title(f"Histogram of Oriented Gradients: {distance_sorted[1]}")
ax4.imshow(dict_hog[distance_sorted[2]], cmap=plt.cm.gray)
ax4.set_title(f"Histogram of Oriented Gradients: {distance_sorted[2]}")

plt.show()

#%%

######################################## SEGMENTATION METHOD ########################################
print("\n")
print("3. Segmentation method \n")

# Segment the input image
s_newImage = segmentation_threshold(newImage)


# Define a dictionary to store the segmentations
dict_s = {}

# Create a loop to iterate the list of images
for img in dict_thres:
    # Threshold segmentation of all images
    s = segmentation_threshold(dict_thres[img])
    # Store the result in the dictionary
    dict_s[img] = s

# A) Compare with euclidean distance for every combination
# Define a dictionary to store the euclidean distances
distance_array_seg={}
# Create a loop to iterate the list of hog images
for s in dict_s:
    # Calculate the euclidean distance of the input image with the ones of the data base
    distance = cv2.norm(s_newImage, dict_s[s])
    # Store the names and the distances of the images in the dictionary
    distance_array_seg[s]=distance

# Sort the images by distance
distance_seg_sorted = sorted(distance_array_seg, key=distance_array_seg.get, reverse=False)

# Printing the list of the three images with less distance using loop
for w in distance_seg_sorted:
    print(f"Image name: {w}, Distance: {round(distance_array_seg[w], 4)}")

print("Name of the three images with less distance: ", distance_seg_sorted[0:3])


# Representate the input image and the segmentated images with less distance

fig, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(16, 8), sharex=True, sharey=True) 

# Representate the original input image
ax5.imshow(newImage) 
ax5.set_title('Input image')

# Representate the Hogs of the images with less distances
ax6.imshow(dict_s[distance_seg_sorted[0]])
ax6.set_title(f"Segmentation: {distance_seg_sorted[0]}")
ax7.imshow(dict_s[distance_seg_sorted[1]])
ax7.set_title(f"Segmentation: {distance_seg_sorted[1]}")
ax8.imshow(dict_s[distance_seg_sorted[2]])
ax8.set_title(f"Segmentation: {distance_seg_sorted[2]}")

plt.show()
