import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import glob
import os
from PIL import Image
import tensorflow
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils


from astropy.visualization import (MinMaxInterval, SqrtStretch,
                                   ImageNormalize)


Path = "../static/Image/testPicture/"
Input = "../static/Image/DataNewTrain/"
filename=".jpg"
import numpy as np
import cv2
import time
cap = cv2.VideoCapture(0)
cap.set(3,1280)

cap.set(4,1024)
h = 1280
w = 1024
Count = 1
SET = 356
center = (w/2,h/2)
scale = 1
M = cv2.getRotationMatrix2D(center,180,scale)
x=50
y=50
while(True):
    # Capture frame-by-frame
	ret, frame = cap.read()
	frame =cv2.transpose(frame,frame);
	small = cv2.resize(frame, (512,640))
	cv2.imshow('IDS_camera',small)
	key = cv2.waitKey(1)
	if key == ord('s'):
		cv2.destroyAllWindows()
		ret2, frame2 = cap.read()
		frame2 =cv2.transpose(frame2,frame2);
		small2 = cv2.resize(frame2, (512,640))
		img = small2.copy()
	elif key == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
height,width,mike = img.shape
mask = np.zeros((height,width), np.uint8)
edges = cv2.Canny(thresh, 100, 200)
cimg=img
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)
for i in circles[0,:]:
    i[2]=i[2]+4
    cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)
masked_data = cv2.bitwise_and(img, img, mask=mask)
_,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
x,y,w,h = cv2.boundingRect(contours[0])
crop = masked_data[y:y+h,x:x+w]
# cv2.imwrite(Path+"Testpic"+filename,crop)
image = cv2.imread("../static/Image/testPicture/Testpic.jpg")
orig = image.copy()

image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

model = load_model("../static/trainmodel/training")

(badlen, goodlen) = model.predict(image)[0]

label = "GOOD" if goodlen> badlen else "BAD"
proba = goodlen if goodlen > badlen else badlen
label = "{}: {:.2f}%".format(label, proba * 100)

output = imutils.resize(orig, width=400)

cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
cv2.imshow("Output", output)
print(label)

cv2.waitKey(0)


cv2.destroyAllWindows()
