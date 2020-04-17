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
    
    
    # Create jacobian of homography matrix
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

def matchFeatures(kpsDsc1, kpsDsc2, debug = False):

    print("FEATURE MATCHING STARTS")
    
    bestDesc1, bestDesc2 = [], []
    minDistance1, minDistance2  = 0, 0
    firstIter = 1
    isSecondDescSet = 0
    
    matches = []
    
    # Find two different points that are closest points to the  soruce point
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
        
    error = cv.norm(p2, predictedPoint2 ,cv.NORM_L2);
    return error


# RANSAC BELOW:
# Two different tresholds, first one is for inliers and second is for termination of the ransac
def ransac(matches, distanceTreshold = 2.5, inlierTreshold = 0.6, debug = False):
    
    print("\nRANSAC STARTS")
    numOfIter = 1000
    bestInliers = []
    bestH = []
    for i in range(numOfIter):
        
        # Get 4 random point matches
        pointPairs = getRandomElements(matches, 4)
        
        # Estimate points based on these matches
        H = calculateHomography(pointPairs)
        
        inliers = []
        for match in matches:
            # If threshold is lower than threshold, regard match as inlier
            distance = calculateEuclidianDistance(match, H)
            if(distance < distanceTreshold):
                inliers.append(match)
        
        # Update inlier and homography
        if len(inliers) > len(bestInliers):
            bestH = H
            bestInliers = inliers
            
        if debug:
            print("Found inliers --> " + str(len(inliers)) + " Max inliers --> " + str(len(bestInliers)))        
       
        # If enoguh number of inliers are found, stop process
        if len(bestInliers) > (len(matches) * inlierTreshold):
            break
        
    print("Final homography --> \n" + str(bestH))
    
    # Save homography in text file
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

    out = np.zeros((rows1 + rows2, cols1 + cols1 + cols2), dtype='uint8')

    # create indices of the destination image and linearize them
    h, w = img1.shape[:2]
    indy, indx = np.indices((h, w), dtype=np.float32)
    lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])
    
    # warp the coordinates of src to those of true_dst
    map_ind = H.dot(lin_homg_ind)
    map_x, map_y = map_ind[:-1]/map_ind[-1]  # ensure homogeneity
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)
    
    # remap pixels from image 1 to image 2
    dst = cv.remap(img2, map_x, map_y, cv.INTER_LINEAR)
    blended = cv.addWeighted(img1, 0.7, dst, 0.3, 0)
    cv.imwrite('stitchedImage.png', blended)
            
    return out

# Not working
def doGaussianNewton(bestInliers, initialH):
    print("GAUS NEWTON STARTS")
    bestH = initialH
    for i in range(15):
        
        jacobian = np.matrix([0,0,0,0,0,0,0,0])
        res = np.matrix([0])
        # For all inliers estimate delta homography
        for inlier in bestInliers:
            
            sensedPoint = inlier[1]
            
            predictedPoint = np.matmul(bestH, np.vstack([np.matrix(inlier[0]).T, [1]]))
            predictedPoint = predictedPoint / predictedPoint[2]
            predictedPoint = predictedPoint[0:2]
            
            # Contruct jacobian of homography matrix
            D = 1/(bestH[2,0] * bestH[0,0] + bestH[2,1] * inlier[0][1] + 1)
            jacobian = np.vstack([jacobian , np.multiply(D , [inlier[0][0], inlier[0][1], 1, 0, 0, 0, -inlier[0][0] * predictedPoint[0,0], -inlier[0][1] * predictedPoint[0,0]])])
            jacobian = np.vstack([jacobian , np.multiply(D , [0, 0, 0, inlier[0][0], inlier[0][1], 1, -inlier[0][0] * predictedPoint[1,0], -inlier[0][1] * predictedPoint[1,0]])])
                
            
            res = np.vstack([res , [inlier[1][0] - predictedPoint[0,0] ]])
            res = np.vstack([res , [inlier[1][1] - predictedPoint[1,0] ]])
        
            
           
            
        jacobian = jacobian[1:, :]
        res = res[1:, :]
        # Calculate delta homogtaphy
        deltaH = np.matmul(np.linalg.inv(np.matmul(jacobian.T, jacobian)), np.matmul(jacobian.T, res))
        deltaH = np.vstack([deltaH, [1]])
        deltaH = deltaH.reshape((3,3))
         
        # Find new homography
        bestH = bestH - deltaH
        
        # Find error based on new homography
        error = 0
        for inlier in bestInliers:
            
            predictedPointForTest = np.matmul(bestH, np.vstack([np.matrix(inlier[0]).T, [1]]))
            predictedPointForTest = predictedPointForTest / predictedPointForTest[2]
            predictedPointForTest = predictedPointForTest[0:2]
            
            
            errorDistance = cv.norm(np.array(inlier[1]), predictedPointForTest, cv.NORM_L2)
            error += errorDistance 
        print("Error --> " + str(error))
        
    return bestH

def drawMatches(img1, img2, matches):
    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1])

    # Place the second image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # x - columns, y - rows
        (x1,y1) = mat[0]
        (x2,y2) = mat[1]

        cv.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        cv.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)

    return out

def saveMatchesInImage(img1, img2, matches, imageName):
    
    outImg = drawMatches(img1, img2, matches)
    cv.imwrite(imageName, outImg)

rootPath = "./data_image_stitching/"

# Read images
image1, image1Gray = loadImages(rootPath, "im89.jpg")
image2, image2Gray = loadImages(rootPath, "im90.jpg")

# Find keypoints and descriptors
kpsDsc1 = extractKeypointsAndDescriptors(image1Gray)
kpsDsc2 = extractKeypointsAndDescriptors(image2Gray)

print("Found keypoints in img1 --> " + str(len(kpsDsc1["keypoints"])))
print("Found keypoints in img2 --> " + str(len(kpsDsc2["keypoints"])))

# Run brute force matcher to find matched points
matches = matchFeatures(kpsDsc1, kpsDsc2, False)
saveMatchesInImage(image1, image2, matches, 'Matches.png')

# Run RANSAC to eliminate wrong matches and find initial homography
initialH, inliers = ransac(matches, debug = True)
# finalH = doGaussianNewton(inliers, initialH)
saveMatchesInImage(image1, image2, inliers, 'MatchesInliers.png')

# Warp images according to homography
warpImages(image1Gray, image2Gray, initialH)

