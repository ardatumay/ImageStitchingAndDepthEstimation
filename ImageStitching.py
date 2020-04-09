# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 00:36:51 2020

@author: arda1
"""
import random 
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
        
    return {"keypoints": keypointsInFloat, "descriptors": descriptors}


def getRandomElements(_list, size):
    
    indexes = random.sample(range(0, len(_list)), size)
    elements = []
    for i in indexes:
        elements.append(_list[i])
    
    return elements
    
def calculateHomography(points):
    
    jacobian = np.matrix([0,0,0,0,0,0,0,0])
    for match in points:
        jacobian = np.vstack([jacobian , [match[0][0], match[0][1], 1, 0, 0, 0, -match[0][0] * match[1][0], -match[0][1] * match[1][0]]])
        jacobian = np.vstack([jacobian , [0, 0, 0, match[0][0], match[0][1], 1, -match[0][0] * match[1][1], -match[0][1] * match[1][1]]])
    jacobian = jacobian[1:, :]
    
    res = np.matrix([0])
    for match in points:
        res = np.vstack([res , [match[1][0]]])
        res = np.vstack([res , [match[1][1]]])
    res = res[1:, :]
    
    H = np.matmul(np.linalg.inv(np.matmul(jacobian.T, jacobian)), np.matmul(jacobian.T, res))
    H = np.vstack([H, [1]])
    H = H.reshape((3,3))
    
    return H;   

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

matches = []
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
            minDistance2 = minDistance1
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
        # print(indexDesc1)
        x = kpsDsc1["keypoints"][i]
        y = kpsDsc2["keypoints"][indexDesc1]
        matches.append((x, y))
    
    isSecondDescSet = 0
    firstIter = 1
    j = 0
    i += 1
    # print(i)




# RANSAC BELOW
numOfIter = 2000
distanceTreshold = 36
bestNumOfInliers = 0
bestInliers = []
bestH = []
for i in range(numOfIter):
    
    pointPairs = getRandomElements(matches, 4)
    
    H = calculateHomography(pointPairs)
    
    numOfInliers = 0
    inliers = []
    for match in matches:
        predictedPoint1 = np.matmul(H, np.vstack([np.matrix(match[0]).T, [1]]))
        predictedPoint1 = predictedPoint1 / predictedPoint1[2]
        predictedPoint1 = predictedPoint1[0:2]
        # predictedPoint2x = np.divide((np.multiply(H[0, 0], matches[0][0][0]) + np.multiply(H[0, 1] , matches[0][0][1]) + H[0, 2]) ,  (np.multiply(H[2, 0], matches[0][0][0]) + np.multiply(H[2, 1], matches[0][0][1]) + 1 ))
        # predictedPoint2y = np.divide((np.multiply(H[1, 0], matches[0][0][0]) + np.multiply(H[1, 1] , matches[0][0][1]) + H[1, 2]) ,  (np.multiply(H[2, 0], matches[0][0][0]) + np.multiply(H[2, 1], matches[0][0][1]) + 1 )) 
        
        distance = L2Norm(np.array(match[1]).T, (predictedPoint1.T))
        if(distance < distanceTreshold):
            numOfInliers += 1
            inliers.append(match)
    
    if numOfInliers > bestNumOfInliers:
        bestNumOfInliers = numOfInliers
        bestH = H
        bestInliers = inliers
        
    print(bestNumOfInliers )




#kullandığın paketlerin version ını ver

# sift için opencv 3.4.1 kullanılma, üst versiyonlarda sift patentli ve kullanılamıyor