import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import argparse
import glob
import os
from random import shuffle

IMAGE_SIZE = 50
BLOCKSIZE = 20
BLOCKSTRIBE = BLOCKSIZE//2
CELLSIZE = BLOCKSIZE
NBINS = 9
DERIVAPERTURE = 1
WINSIGMA = -1.
HISTOGRAMNORMTYPE = 0
L2HYSTHRESHOLD = 0.2
GAMMACORRECTION = 1
NLEVELS = 64
SINEDGRADIENTS = True

DATA_SET_NAME = "HOGFull1Font"
FILENAME = "{}-imagesize-{}-block-{}-cell-{}-bin-{}-sined-{}"
FILENAME = FILENAME.format(	DATA_SET_NAME,
							IMAGE_SIZE,
							BLOCKSIZE,
							CELLSIZE,
							NBINS,
							SINEDGRADIENTS)


def createHOGDescription():
	winSize = 			(IMAGE_SIZE,	IMAGE_SIZE)
	blockSize = 		(BLOCKSIZE, 	BLOCKSIZE)
	blockStride = 		(BLOCKSTRIBE, 	BLOCKSTRIBE)
	cellSize = 			(CELLSIZE,		CELLSIZE)
	nbins = 			NBINS
	derivAperture = 	DERIVAPERTURE
	winSigma = 			WINSIGMA
	histogramNormType = HISTOGRAMNORMTYPE
	L2HysThreshold = 	L2HYSTHRESHOLD
	gammaCorrection = 	GAMMACORRECTION
	nlevels = 			NLEVELS
	signedGradients = 	SINEDGRADIENTS

	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
	                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
	return hog

path = "C:\Python_program\Machine_learning\images\SV_BL\\"
name = "SEMI_SV_B4_BL_001"
img = cv2.imread(path+name+".jpg",0)
img1 = cv2.imread(path+name+".jpg")

# img = cv2.resize(img,(100,100))
kernel = np.ones((5,5),np.float32)
kernel_3 = np.ones((3,3),np.float32)/9
kernel_chud= np.array([[0, -1, 0],
					[-1, 5, -1],
					[0, -1, 0]], np.float32)

# kernel_chud2= np.array([[-1, -1, -1, -1, -1],
# 					[-1, -1, -1, -1, -1],
# 					[-1, -1, 25 -1, -1],
# 					[-1, -1, -1, -1, -1],
# 					[-1, -1, -1, -1, -1]], np.float32)
laplacian_kernel = np.array(
		[[-1,-1,-1,-1,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1],	
		[-1,-1,-1,-1,-1,-1,-1],
		[-1,-1,-1,48,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1]], dtype='int')

laplacian_kernel2 = np.array((
		[-1,-1,-1,-1,-1,-1,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1,-1,-1],	
		[-1,-1,-1,-1,-1,-1,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1,-1,-1],
		[-1,-1,-1,-1,80,-1,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1,-1,-1]), dtype='int')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
height,width = img.shape

mask = np.zeros((height,width), np.uint8)
edges = cv2.Canny(thresh, 100, 200)
cimg=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)
for i in circles[0,:]:
    i[2]=i[2]+4
    # Draw on mask
    cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)

masked_data = cv2.bitwise_and(img, img, mask=mask)

_,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
x,y,w,h = cv2.boundingRect(contours[0])
crop = masked_data[y:y+h,x:x+w]

blur = cv2.GaussianBlur(crop,(5,5),0)
hog = createHOGDescription()

# dst = cv2.filter2D(crop,-1,kernel)
# dst_2 = cv2.filter2D(crop,-1,kernel_chud)


# laplacian = cv2.Laplacian(dst_2,cv2.CV_64F,ksize=11)
# laplacian_dst = cv2.Laplacian(dst_2,cv2.CV_8U,ksize=7)

Laplaceimage=  cv2.filter2D(blur,-1,laplacian_kernel2)
# aoey = cv2.Laplacian(dst_2,cv2.CV_8U,ksize=11)
# laplacian_blur =  cv2.filter2D(aoey,-1,kernel_3)
Laplaceimage_blur = cv2.blur(Laplaceimage,(9,9))
ret,thresh2 = cv2.threshold(Laplaceimage_blur,127,255,cv2.THRESH_BINARY)
thresh2_blur =  cv2.filter2D(thresh2,-1,kernel)
kernel_erode = np.ones((5,5),np.uint8)
erosion = cv2.erode(thresh2_blur,kernel_erode,iterations = 1)
cv2.imwrite( name +"_PREPRO.jpg",erosion)
cv2.imshow("erosionImage",erosion)
cv2.imshow("LaplaceIMAGE",Laplaceimage_blur)
cv2.imshow("thEressh",thresh2_blur)
print(hog.compute(Laplaceimage_blur))
cv2.waitKey(0)
cv2.destroyAllWindows()
	
