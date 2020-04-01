# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 00:36:51 2020

@author: arda1
"""

import numpy as np
import cv2 as cv

rootPath = "./data_image_stitching/"

def loadImagesToBeStitched(filePath, image1, image2):

    img1 = cv.imread(filePath + image1)
    grayImg1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)    
    
    img2 = cv.imread(filePath + image2)
    grayImg2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)    
    
    return grayImg1, grayImg2


image1, image2 = loadImagesToBeStitched(rootPath, "im1.png", "im2.png")

sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(image1, None)

img=cv.drawKeypoints(image1, kp, image1)
cv.imwrite('sift_keypoints.jpg',img)