import cv2
import numpy as np
import math
from DescriptorComputer import DescriptorComputer

class EdgeHistogramComputer(DescriptorComputer):

	def __init__(self, rows, cols):
		sqrt2 = math.sqrt(2)
		self.kernels = (np.matrix([[1,1],[-1,-1]]), \
				np.matrix([[1,-1],[1,-1]]),         \
				np.matrix([[sqrt2,0],[0,-sqrt2]]),  \
				np.matrix([[0,sqrt2],[-sqrt2,0]]),  \
				np.matrix([[2,-2],[-2,2]]));
		self.bins = [len(self.kernels)]
		self.range = [0,len(self.kernels)]
		self.rows = rows
		self.cols = cols
		self.prefix = "EDH"
		
	def compute(self, frame):
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		descriptor = []
		dominantGradients = np.zeros_like(frame)
		maxGradient = cv2.filter2D(frame, cv2.CV_32F, self.kernels[0])
		maxGradient = np.absolute(maxGradient)
		for k in range(1,len(self.kernels)):
			kernel = self.kernels[k]
			gradient = cv2.filter2D(frame, cv2.CV_32F, kernel)
			gradient = np.absolute(gradient)
			np.maximum(maxGradient, gradient, maxGradient)
			indices = (maxGradient == gradient)
			dominantGradients[indices] = k
		
		frameH, frameW = frame.shape
		for row in range(self.rows):
			for col in range(self.cols):
				mask = np.zeros_like(frame)
				mask[((frameH/self.rows)*row):((frameH/self.rows)*(row+1)),(frameW/self.cols)*col:((frameW/self.cols)*(col+1))] = 255
				hist = cv2.calcHist([dominantGradients], [0], mask, self.bins, self.range)
				hist = cv2.normalize(hist, None)
				descriptor.append(hist)
		return np.concatenate([x for x in descriptor])
		
if __name__ == "__main__":
	path = r"C:\Python_program\Machine_learning\CNN_model\Allimage\B4\newbad\Base_4_BL_8_7.jpg"
	img = cv2.imread(path)
    Edge = EdgeHistogramComputer(2,2)
    img2 = Edge.compute(img)
    cv2.imshow("comput", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()