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
		for i in range(0,11):
			time.sleep(0.25)
			print("Capture"+str(Count)+"__"+str(SET))
			if Count == 11 :
				SET +=1
				Count =0
			ret2, frame2 = cap.read()
			frame2 =cv2.transpose(frame2,frame2);
			small2 = cv2.resize(frame2, (512,640)) 
			cv2.imwrite("B4/Base_4_BL_"+str(SET)+"_"+str(Count)+str(".jpg"),small2)
			Count +=1
		print ("Finish")
	elif key == ord('q'): 
		break 
cap.release()
cv2.destroyAllWindows()