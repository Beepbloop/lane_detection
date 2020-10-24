# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 19:16:56 2020

@author: prana
"""


import cv2 as cv
import numpy as np
from PIL import Image
from colorsys import rgb_to_hsv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# Read image from JPEG file
pil_im = Image.open("lane_sample1.jpeg")
orig = np.array(pil_im)

# Resize so that standard height is 512
width = int(pil_im.width * 512 / pil_im.height)
pil_im = pil_im.resize((width , 512))

# Convert to np array
orig_img = np.array(pil_im)
plt.imshow(orig_img)
plt.axis('off')
plt.show()

'''
def RGB_to_HSV(rgb):
    
    rows = rgb.shape[0]
    cols = rgb.shape[1]
    hsvImage = np.zeros((rows, cols))
    
    for x in range(rows):
        for y in range(cols):
            x = 1
    
    return hsvImage

imHSV = RGB_to_HSV(orig_img)
'''
rows = orig_img.shape[0]
cols = orig_img.shape[1]
hsvImage = np.zeros((rows, cols, 3))

for x in range(rows):
    for y in range(cols):
        redPrime = orig_img[x,y,0] / 255
        greenPrime = orig_img[x,y,1] / 255
        bluePrime = orig_img[x,y,2] / 255
        cMax = max(redPrime, greenPrime, bluePrime)
        cMin = min(redPrime, greenPrime, bluePrime)
        delta = cMax - cMin

        #hue
        if delta == 0:
            hsvImage[x][y][0] = 0
        elif cMax == redPrime:
            hsvImage[x][y][0] = (60 * ((greenPrime - bluePrime) / delta) + 360) % 360
        elif cMax == greenPrime:
            hsvImage[x][y][0] = (60 * ((bluePrime - redPrime) / delta) + 120) % 360
        elif cMax == bluePrime:
            hsvImage[x][y][0] = (60 * ((redPrime - greenPrime) / delta) + 240) % 360
        
        #saturation
        if cMax == 0:
            hsvImage[x][y][1] = 0
        else:
            hsvImage[x][y][1] = (delta / cMax)
        
        #value
        hsvImage[x][y][2] = cMax

plt.imshow(hsvImage, cmap = 'hsv')
plt.axis('off')
plt.show()

