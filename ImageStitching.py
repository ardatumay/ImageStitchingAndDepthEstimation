# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 00:36:51 2020

@author: arda1
"""
import random 
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


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

def matchFeaturesBuiltin(kp1, kp2, desc1, desc2):
    print("Matching Features...")
    matcher = cv.BFMatcher(cv.NORM_L2, True)
    matches = matcher.match(desc1, desc2)
    return matches

def matchFeatures(kpsDsc1, kpsDsc2, debug = False):

    print("FEATURE MATCHING STARTS")
    
    bestDesc1, bestDesc2 = [], []
    minDistance1, minDistance2  = 0, 0
    firstIter = 1
    isSecondDescSet = 0
    
    matches = []
    
    i, j = 0, 0
    indexDesc2 =  0
    for descriptor1 in kpsDsc1["descriptors"]:
        
        for descriptor2 in kpsDsc2["descriptors"]:
            
            distance = cv.norm(descriptor1, descriptor2 ,cv.NORM_L2);
            
            if isSecondDescSet == 1 and distance < minDistance2 and distance > minDistance1:
                bestDesc2 = descriptor2
                minDistance2 = distance
                
            if firstIter == 1:
                minDistance1 = distance
                bestDesc1 = descriptor2
                indexDesc2 = j
                
            elif distance <= minDistance1:
                bestDesc1 = descriptor2
                minDistance2 = minDistance1
                minDistance1 = distance
                indexDesc2 = j
                
            elif isSecondDescSet == 0:
                bestDesc2 = descriptor2
                minDistance2 = distance
                isSecondDescSet = 1
           
            firstIter = 0
            j += 1
            # print(minDistance1, "   ", j)
        
        if minDistance1 < 0.75 * minDistance2:
            # print(indexDesc1)
            x = kpsDsc1["keypoints"][i]
            y = kpsDsc2["keypoints"][indexDesc2]
            matches.append((x, y))
            
            if debug:
                print("{:.0f} Point 1 --> {:.2f}, {:.2f} Point 2 --> {:.2f}, {:.2f} Distance --> {:.2f}".format(len(matches), x[0], x[1], y[0], y[1], minDistance1))
                
        isSecondDescSet = 0
        firstIter = 1
        j = 0
        i += 1
        
    return matches

# Calculate error between predicted and real points in image to by euclidian distance
def calculateEuclidianDistance(pointPair, H):
    
    # predictedPoint2 = np.dot(H, np.vstack([np.matrix(match[0]).T, [1]]))
    predictedPoint2 = np.matmul(H, np.vstack([np.matrix(pointPair[0]).T, [1]]))
    predictedPoint2 = predictedPoint2 / predictedPoint2[2]
    predictedPoint2 = predictedPoint2[0:2]
    
    p2 = pointPair[1]
    # predictedPoint2x = np.divide((np.multiply(H[0, 0], matches[0][0][0]) + np.multiply(H[0, 1] , matches[0][0][1]) + H[0, 2]) ,  (np.multiply(H[2, 0], matches[0][0][0]) + np.multiply(H[2, 1], matches[0][0][1]) + 1 ))
    # predictedPoint2y = np.divide((np.multiply(H[1, 0], matches[0][0][0]) + np.multiply(H[1, 1] , matches[0][0][1]) + H[1, 2]) ,  (np.multiply(H[2, 0], matches[0][0][0]) + np.multiply(H[2, 1], matches[0][0][1]) + 1 )) 
        
    error = np.matrix([[p2[0]], [p2[1]]]) - np.array(predictedPoint2)
    return np.linalg.norm(error)
    
    # error = cv.norm(p2, predictedPoint2 ,cv.NORM_L2);
    # return error


# RANSAC BELOW:
# Two different tresholds, first one is for inliers and second is for termination of the ransac
def ransac(matches, distanceTreshold = 2.5, inlierTreshold = 0.6, debug = False):
    
    print("\nRANSAC STARTS")
    numOfIter = 1000
    bestInliers = []
    bestH = []
    for i in range(numOfIter):
        
        pointPairs = getRandomElements(matches, 4)
        
        H = calculateHomography(pointPairs)
        
        inliers = []
        for match in matches:
            distance = calculateEuclidianDistance(match, H)
            if(distance < distanceTreshold):
                inliers.append(match)
        
        if len(inliers) > len(bestInliers):
            bestH = H
            bestInliers = inliers
            
        if debug:
            print("Found inliers --> " + str(len(inliers)) + " Max inliers --> " + str(len(bestInliers)))        
       
        if len(bestInliers) > (len(matches) * inlierTreshold):
            break
        
    print("Final homography --> \n" + str(bestH))
    
    f = open('homography.txt', 'w')
    f.write("Final homography: \n" + str(bestH)+"\n")
    f.write("Final inliers count: " + str(len(bestInliers)))
    f.close()
        
    return bestH, bestInliers

def warpImages(img1, img2, H):
    
        
        
    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    H = np.matrix([[ 1.15976269e+00, -2.34391473e-02, -6.46162755e+02],
                   [2.83148807e-03,  1.12860047e+00,  1.38407135e+02],
                   [1.52419846e-04,  1.37118880e-05,  1.00000000e+00]])
    

    out = cv.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0] +500))
    out[0:img2.shape[0], 400:  400 +img2.shape[1]] = img2

    # Place the first image to the left
    # out[:rows1,:cols1,:] = np.dstack([img1])

    # Place the next image to the right of it
    # out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2])
    
    # rows = img1.shape[0]
    # cols = img1.shape[1] 
    
    # for row in range(rows):
    #     for col in range(cols):
    #         point = np.matrix([row, col, 1])
    #         predictedPoint = np.matmul(H, point.T)
    #         predictedPoint = predictedPoint / predictedPoint[2]
    #         predictedPoint = predictedPoint[0:2]
    #         out[int(predictedPoint[0, 0]), int(predictedPoint[1, 0])] = img2[row, col]
    
    

    return out

rootPath = "./data_image_stitching/"

image1, image1Gray = loadImages(rootPath, "im1.png")
image2, image2Gray = loadImages(rootPath, "im2.png")

kpsDsc1 = extractKeypointsAndDescriptors(image1Gray)
kpsDsc2 = extractKeypointsAndDescriptors(image2Gray)

print("Found keypoints in img1 --> " + str(len(kpsDsc1["keypoints"])))
print("Found keypoints in img2 --> " + str(len(kpsDsc2["keypoints"])))

matches = matchFeatures(kpsDsc1, kpsDsc2, True)
# matches = matchFeaturesBuiltin(kpsDsc1["keypoints"], kpsDsc2["keypoints"], kpsDsc1["descriptors"], kpsDsc2["descriptors"])

finalH, inliers = ransac(matches, debug = True)

# matchImg = warpImages(image1Gray, image2Gray, None)
# cv.imwrite('Matches.png', matchImg)


print("GAUS NEWTON STARTS")

# error = 0
# for inlier in bestInliers:
    
#     predictedPointForTest = np.matmul(bestH, np.vstack([np.matrix(inlier[0]).T, [1]]))
#     predictedPointForTest = predictedPointForTest / predictedPointForTest[2]
#     predictedPointForTest = predictedPointForTest[0:2]
    
#     errorDistance = L2Norm(np.array(inlier[1]).T, (predictedPointForTest.T))
#     error += errorDistance 
# print("Error --> " + str(error))
    

# while 500:
#     jacobian = np.matrix([0,0,0,0,0,0,0,0])
#     res = np.matrix([0])
#     for inlier in bestInliers:
        
#         sensedPoint = inlier[1]
        
#         predictedPoint = np.matmul(bestH, np.vstack([np.matrix(inlier[0]).T, [1]]))
#         predictedPoint = predictedPoint / predictedPoint[2]
#         predictedPoint = predictedPoint[0:2]
        
#         D = 1/(bestH[2,0] * inlier[0][0] + bestH[2,1] * inlier[0][1] + 1)
#         jacobian = np.vstack([jacobian , np.multiply(D , [inlier[0][0], inlier[0][1], 1, 0, 0, 0, -inlier[0][0] * predictedPoint[0,0], -inlier[0][1] * predictedPoint[0,0]])])
#         jacobian = np.vstack([jacobian , np.multiply(D , [0, 0, 0, inlier[0][0], inlier[0][1], 1, -inlier[0][0] * predictedPoint[1,0], -inlier[0][1] * predictedPoint[1,0]])])
            
        
#         res = np.vstack([res , [inlier[1][0] - predictedPoint[0,0] ]])
#         res = np.vstack([res , [inlier[1][1] - predictedPoint[1,0] ]])
    
        
       
        
#     jacobian = jacobian[1:, :]
#     res = res[1:, :]
    
#     deltaH = np.matmul(np.linalg.inv(np.matmul(jacobian.T, jacobian)), np.matmul(jacobian.T, res))
#     deltaH = np.vstack([deltaH, [1]])
#     deltaH = deltaH.reshape((3,3))
     
#     bestH = bestH + deltaH
    
    
#     error = 0
#     for inlier in bestInliers:
        
#         predictedPointForTest = np.matmul(bestH, np.vstack([np.matrix(inlier[0]).T, [1]]))
#         predictedPointForTest = predictedPointForTest / predictedPointForTest[2]
#         predictedPointForTest = predictedPointForTest[0:2]
        
#         errorDistance = L2Norm(np.array(inlier[1]).T, (predictedPointForTest.T))
#         error += errorDistance 
#     print("Error --> " + str(error))

# dst = cv.warpPerspective(image1,bestH,(image2.shape[1] + image1.shape[1], image2.shape[0]))
# plt.imshow(dst)
# plt.show()
#kullandığın paketlerin version ını ver

# sift için opencv 3.4.1 kullanılma, üst versiyonlarda sift patentli ve kullanılamıyor