# from scipy import ndimage, misc
# import matplotlib.pyplot as plt
# from scipy.misc import imread
# from skimage.color import rgb2gray
# def any_neighbor_zero(img, i, j):
#     for k in range(-1,2):
#       for l in range(-1,2):
#          if img[i+k, j+k] == 0:
#             return True
#     return False


# def zero_crossing(img):
# 	img[img > 0] = 1
# 	img[img  0 ]	and any_neighbor_zero(img, i, j):
# 		out_img[i,j] = 255
# 	return out_img


# img = rgb2gray(imread('./lencrop/Base_4_BL_1_2.jpg'))

# fig = plt.figure(figsize=(25,15))
# plt.gray() # show the filtered result in grayscale
 
# for sigma in range(2,10, 2):
# plt.subplot(2,2,sigma/2)
# result = ndimage.gaussian_laplace(img, sigma=sigma)
# plt.imshow(zero_crossing(result))
# plt.axis('off')
# plt.title('LoG with zero-crossing, sigma=' + str(sigma), size=30)
 
# plt.tight_layout()
# plt.show()

import scipy as sp
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import cv2
from scipy.misc import imread
from skimage.color import rgb2gray

columns =2 
rows =1
lena = rgb2gray(imread('./lencrop/Base_4_BL_19_4.jpg'))

# lena = sp.misc.lena()
LoG = nd.gaussian_laplace(lena, 3)

thres = np.absolute(LoG).mean() *0.75
output = sp.zeros(LoG.shape)
w = output.shape[1]
h = output.shape[0]

for y in range(1, h - 1):
    for x in range(1, w - 1):
        patch = LoG[y-1:y+2, x-1:x+2]
        p = LoG[y, x]
        maxP = patch.max()
        minP = patch.min()
        if (p > 0):
            zeroCross = True if minP < 0 else False
        else:
            zeroCross = True if maxP > 0 else False
        if ((maxP - minP) > thres) and zeroCross:
            output[y, x] = 1

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(LoG)
fig.add_subplot(1,2,2)
plt.imshow(output)
plt.show()