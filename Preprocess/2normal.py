import cv2
import numpy
 
path = r"C:\Python_program\Machine_learning\CNN_model\Allimage\B4\newbad\Base_4_BL_8_7.jpg"
img = cv2.imread(path)
img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
 
 
cv2.imwrite('result.jpg',hist_equalization_result)
cv2.imwrite('resul2t.jpg',img)