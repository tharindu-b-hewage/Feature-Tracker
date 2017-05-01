import cv2
import numpy as np

def HarrisCorner(src, alpha): #src: grayscale image with cv_64F
    Ix = cv2.Sobel(src,-1,1,0,ksize=5)
    Iy = cv2.Sobel(src,-1,0,1,ksize=5)
    Ix2 = cv2.GaussianBlur(np.multiply(Ix,Ix),(5,5),5,sigmaY=5)
    Iy2 = cv2.GaussianBlur(np.multiply(Iy,Iy),(5,5),5,sigmaY=5)
    IxIy = cv2.GaussianBlur(np.multiply(Ix,Iy),(5,5),5,sigmaY=5)
    harrisResponce = np.zeros(src.shape, dtype="float64")
    #print src.shape[0]
    #input("")
    for x in xrange(2,src.shape[1]-1):
        for y in xrange(2,src.shape[0]-1):
            #print "convolving harris: "+str(x) + "," + str(y) 
            sum_Ix2 = np.sum(Ix2[y-2:y+2, x-2:x+2])
            sum_Iy2 = np.sum(Iy2[y-2:y+2, x-2:x+2])
            sum_IxIy = np.sum(IxIy[y-2:y+2, x-2:x+2])
            detM = sum_Ix2 * sum_Iy2 - sum_IxIy * sum_IxIy
            traceM = sum_Ix2 + sum_Iy2
            harrisResponce[y,x] = detM - alpha * traceM * traceM
    maxR = np.max(harrisResponce)
    minR = np.min(harrisResponce)
    addingPart = minR * np.ones(harrisResponce.shape())
    divider = maxR - minR
    np.subtract(harrisResponce, addingPart)
    np.divide(harrisResponce, divider)
    return harrisResponce

def nonMaximalSupress1(image,NHoodSize):
    #
    dX, dY = NHoodSize
    M, N = image.shape
    for x in range(0,M-dX+1):
        print M-dX+1-x
        for y in range(0,N-dY+1):
            window = image[x:x+dX, y:y+dY]
            if np.sum(window)==0:
                localMax=0
            else:
                localMax = np.amax(window)
            maxCoord=np.unravel_index(np.argmax(window), window.shape) + np.array((x,y))
            #suppress everything
            image[x:x+dX, y:y+dY]=0
            #reset only the max
            if localMax > 0:
                #print localMax
                #print "max coord is ", maxCoord
                image[tuple(maxCoord)] = localMax
    return image

def nonMaximaSupression(corner_image_gray):
    positiveImage = corner_image_gray.copy()
    