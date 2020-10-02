from GrayHistogramComputer import GrayHistogramComputer
import cv2
path = r"C:\Python_program\Machine_learning\CNN_model\Allimage\B4\newbad\Base_4_BL_8_7.jpg"
img = cv2.imread(path,1)
print(img.shape)
computer = GrayHistogramComputer(361,361,2)
descriptor = computer.compute(img) #so descriptor is a vector of 2 x 2 x 32 dimensions
cv2.imshow("ASDSAD",descriptor)
cv2.waitKey(0)
cv2.destroyAllWindows()