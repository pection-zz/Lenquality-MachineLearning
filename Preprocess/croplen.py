import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt


from astropy.visualization import (MinMaxInterval, SqrtStretch,
                                   ImageNormalize)
kernel = np.ones((5,5),np.float32)/25
kernel_3 = np.ones((3,3),np.float32)/9
kernel_chud= np.array([[0, -1, 0],
					[-1, 5, -1],
					[0, -1, 0]], np.float32)
kernel_erode = np.ones((5,5),np.uint8)

laplacian_kernel = np.array((
		[-1,-1,-1,-1,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1],
		[-1,-1,-1,48,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1]), dtype=np.float32)
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
# img1 = cv2.imread("./images/bad/SEMI_SV_B2_BC_01_1.jpg")
# img = cv2.imread("./images/bad/SEMI_SV_B2_BC_01_1.jpg",0)
# img1 = cv2.imread("./images/bad/SEMI_SV_B1_BL_01_1.jpg")
# img = cv2.imread("./images/bad/SEMI_SV_B1_BL_01_1.jpg",0)	

ap = argparse.ArgumentParser()
ap.add_argument("-i","--input",required = True, help="IMAGE input")
ap.add_argument("-o","--output",required = True, help="IMAGE output")

args = vars(ap.parse_args())

fileList = glob.glob(os.path.abspath(args["input"])+"/*.*")

Path = args["output"]+"/"
numberpicture = 0
for name in fileList:
	assert name is not None
	img = cv2.imread(name)
	img1 = cv2.imread(name,0)
	filename = (os.path.basename(name))



	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



	ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
	# print (img.shape)
	height,width,mike = img.shape

	mask = np.zeros((height,width), np.uint8)
	edges = cv2.Canny(thresh, 100, 200)
	cimg=img
	# cv2.imshow("ED",edges)
	# cv2.waitKey(0)
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
	cv2.imshow("CROP",crop)
	cv2.waitKey(0)
	numberpicture = numberpicture +1
	# cv2.imwrite(Path+filename,crop)

cv2.destroyAllWindows()
