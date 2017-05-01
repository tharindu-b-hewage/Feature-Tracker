import cv2
import numpy as np
import functions as F

#filename = "../../cornerTest1.jpg"
filename = "../../hotel.seq0.png"
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = F.HarrisCorner(gray, 0.04)
#dst = cv2.cornerHarris(gray,2,5,0.04)
img1 = img
img[dst>np.amax(dst)*0.01] = [0,0,255]
supressed = F.nonMaximalSupress1(dst,(5,5))
img1[supressed>np.amax(supressed)*0.01] = [0,0,255]
print "dst max = " + str(np.amax(dst))
print  dst[50,50]
#cv2.imshow('dst',dst)
cv2.imshow('image',img)
cv2.imshow('supressed',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()