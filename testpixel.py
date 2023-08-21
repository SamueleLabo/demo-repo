#! python3

"""
 Definition of a uniform emissivity pixel of lunar surface for SurRender
 Script : testpixel
 2023 - Samuele Giuseppe Labò

"""

'''
from PIL import Image
from numpy import array

# Import default texture pixel used by SurRender

im = Image.open("default.png")
out = im.point()
print(im.format, im.size, im.mode)

im = Image.open("SCR02_plotGray.png")
ar = array(im)
print(ar)

ar_g = array(im_g)
im_g = im.convert("L")

'''
import pandas as pd
import numpy as np
from glob import glob
import cv2 
import matplotlib.pylab as plt

# Reading in images
im_set = glob('./Images/*.png') # Return save all .png files in the indicated directory 
print('The test image is:', im_set[2]) 
im_mpl = plt.imread(im_set[2]) # Read image with matplotlib
im_cv2 = cv2.imread(im_set[2]) # Read image with cv2
print(im_mpl.shape,im_cv2.shape) # Height, Width, Channels

'''
imread returns a numpy array with the pixel's intesity (max is usually 255, so dividing by 255 we can normalize)
'''

# Display images
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(im_mpl)
ax.axis('off')
plt.show()

# Display rgb channels of an image
fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].imshow(im_mpl[:,:,0],cmap='Reds')
ax[1].imshow(im_mpl[:,:,1],cmap='Greens')
ax[2].imshow(im_mpl[:,:,2],cmap='Blues')
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
ax[0].set_title('Red Channel')
ax[1].set_title('Green Channel')
ax[2].set_title('Blue Channel')
plt.show()

'''
#cv2 reads BGR while matplot lib reads RGB
'''

# Converting BGR to RGB
im_cv2_rgb = cv2.cvtColor(im_cv2, cv2.COLOR_BGR2RGB)

# Image manipulation
img = plt.imread(im_set[2])
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Converts image from RGB color scale to gray scale
print(img_gray.shape) # Shows that now dimensions are only two since there are no color channels

fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(img_gray, cmap='Greys')
ax.axis('off')
ax.set_title('Grey Image')
plt.show()

# Resizing and Scaling
img_scaled = cv2.resize(img, None, fx=0.25, fy=0.25) # Changes size either specifing desired dimensions (none in this case) or scale factors for each axis 
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(img_scaled, cmap='Greys')
ax.axis('off')
ax.set_title('Grey Image')
plt.show()
print(img_scaled.shape) # By reducing it's like zooming in

img_resized = cv2.resize(img, (100,200)) 
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(img_resized, cmap='Greys')
ax.axis('off')
ax.set_title('Grey Image')
plt.show()
print(img_resized.shape) 


img_resized_up = cv2.resize(img, (5000,5000), interpolation = cv2.INTER_CUBIC) # For upscaling an interpolation model (available ones on documentation) has to be picked so that cv2 knows how to strech pixels out
print(img_resized_up.shape) 

# Sharpening an blurring
'''
#We can apply what are called Kernel to obtain different effects on an image: by looking up cv2 kernels it is possible to see what they do
'''

kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened = cv2.filter2D(img, -1, kernel_sharpening) # Applies to img, at depth -1 (not sure what depth is) the selected kernel or filter

kernel_blurr = np.ones((3,3), np.float32) / 9
blurred = cv2.filter2D(img, -1, kernel_blurr) 

# Save image
plt.imsave('blurred.png', blurred) # Give image name with extension and image
cv2.imwrite('sharpened.png', sharpened) # Same but provides true output when image is saved correctly

# Rotate image
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result, rot_mat

im_rot, rot_mat = rotate_image(im_cv2, 20)
cv2.imshow('Image', im_cv2)
cv2.imshow('Rotated Image', im_rot)
cv2.waitKey(40000)