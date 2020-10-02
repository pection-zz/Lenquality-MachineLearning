import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Python_program\Machine_learning\CNN_model\Allimage\B4\newbad\Base_4_BL_8_7.jpg"
img = cv.imread(path,0)
print(img.shape)
normalizedImg = np.zeros((361, 361))
normalizedImg = cv.normalize(img,  normalizedImg, 0, 255, cv.NORM_MINMAX)
# cv.imshow('dst_rt', normalizedImg)
# cv.imshow('Origin',img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# hist = cv.calcHist([img],[0],None,[256],[0,256])
hist = cv.calcHist([img],[0],None,[256],[0,256])
plt.hist(img.ravel(),256,[0,256])
# plt.ihst(normalizedImg.ravel(),256,[0,256])

plt.title('Histogram for gray scale picture')
plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF     
    if k == 27: break             # ESC key to exit 
cv2.destroyAllWindows()


