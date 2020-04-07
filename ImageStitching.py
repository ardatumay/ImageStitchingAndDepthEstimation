# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 00:36:51 2020

@author: arda1
"""

import numpy as np
import cv2 as cv

rootPath = "./data_image_stitching/"


def L2Norm(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1-vec2)))


def loadImages(filePath, image1):

    image = cv.imread(filePath + image1)
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)    

    return image, grayImage



def extractKeypointsAndDescriptors(image):
    sift = cv.xfeatures2d.SIFT_create()
    
    keypoints, descriptors = sift.detectAndCompute(image, None)
    keypointsInFloat = []
    
    for keypoint in keypoints:
        keypointsInFloat.append(keypoint.pt)
        
    return {"keypoints": keypointsInFloat, "descriptors": descriptors, "rawkeypoints": keypoints}


        

image1, image1Gray = loadImages(rootPath, "im1.png")
image2, image2Gray = loadImages(rootPath, "im2.png")

kpsDsc1 = extractKeypointsAndDescriptors(image1Gray)
kpsDsc2 = extractKeypointsAndDescriptors(image2Gray)

# img=cv.drawKeypoints(image1, kpsDsc1["kp"][0], image1Gray)
# cv.imwrite('sift_keypoints.jpg',img)

bestDesc1, bestDesc2 = [], []
minDistance1, minDistance2  = 0, 0
bestKeypoint1, bestKeypoint2 = [], []
firstIter = 1
isSecondDescSet = 0

mathces = []
distances = []

i, j = 0, 0
indexDesc1 =  0
for descriptor1 in kpsDsc1["descriptors"]:
    
    for descriptor2 in kpsDsc2["descriptors"]:
        
        distance = L2Norm(descriptor1, descriptor2)
        
        if isSecondDescSet == 1 and distance < minDistance2 and distance > minDistance1:
            bestDesc2 = descriptor2
            minDistance2 = distance
            
        if firstIter == 1:
            minDistance1 = distance
            bestDesc1 = descriptor2
            indexDesc1 = j
            
        elif distance <= minDistance1:
            bestDesc1 = descriptor2
            minDistance1 = distance
            indexDesc1 = j
            
        elif isSecondDescSet == 0:
            bestDesc2 = descriptor2
            minDistance2 = distance
            isSecondDescSet = 1
       
        firstIter = 0
        distances.append(minDistance1)
        j += 1
        # print(minDistance1, "   ", j)
    
    if minDistance1 < 0.75 * minDistance2:
        print(indexDesc1)
        x = kpsDsc1["keypoints"][i]
        y = kpsDsc2["keypoints"][indexDesc1]
        mathces.append((x, y))
    
    isSecondDescSet = 0
    firstIter = 1
    j = 0
    i += 1
    # print(i)
    







#kullandığın paketlerin version ını ver

# sift için opencv 3.4.1 kullanılma, üst versiyonlarda sift patentli ve kullanılamıyor