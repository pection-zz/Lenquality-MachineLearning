import cv2 

cam = cv2.VideoCapture(1)

while True:
	ret,img=cam.read()
	cv2.imshow('webcam',img)
	if cv2.waitKey(1) == 27:
		break
cv2.destroyAllWindows()