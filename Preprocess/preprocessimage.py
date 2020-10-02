import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
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
	ret, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
	# print (img.shape)
	height,width,mike = img.shape
	mask = np.zeros((height,width), np.uint8)
	edges = cv2.Canny(img,50, 200)
	cimg=img
	# cv2.imshow("EDGE",edges)
	# cv2.imshow("imgori",img)
	circles = cv2.HoughCircles(img1, cv2.HOUGH_GRADIENT, 1.2, 10000, param1 = 255, param2 = 50, minRadius = 0, maxRadius = 0)
	if circles is not None:
		for i in circles[0,:]:
		    i[2]=i[2]+4
		    cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)
		masked_data = cv2.bitwise_and(img, img, mask=mask)

		# cv2.imshow("masd",masked_data)
		# cv2.waitKey(0)

		_,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

		contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		x,y,w,h = cv2.boundingRect(contours[0])
		crop = masked_data[y:y+h,x:x+w]


		img_to_yuv = cv2.cvtColor(crop,cv2.COLOR_BGR2YUV)
		img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
		hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)



		blur = cv2.GaussianBlur(crop,(5,5),0)
		Laplaceimage=  cv2.filter2D(blur,-1,laplacian_kernel2)
		Laplace_blur = cv2.blur(Laplaceimage,(9,9))
		ret,thresh2 = cv2.threshold(Laplace_blur,100,255,cv2.THRESH_BINARY)
		thresh2_blur =  cv2.GaussianBlur(thresh2,(5,5),0)
		dilation = cv2.erode(thresh2_blur,kernel_erode,iterations = 1)
		normalizedImg = np.zeros((640, 480))
		normalizedImg = cv2.normalize(Laplace_blur,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
		cv2.imwrite(Path+filename,crop)
		numberpicture = numberpicture +1
		cv2.waitKey(0)
cv2.destroyAllWindows()
