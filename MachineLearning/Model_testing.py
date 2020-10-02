
import tensorflow
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2
import glob
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())



fileList = glob.glob(os.path.abspath("/Users/pection/Programing/aboutme/Len-QualityChecking-Using-Machine-Learning/static/Image/"+args["image"])+"/*.*")

numberpicture = 0
for name in fileList:
	assert name is not None
	# print (name)
	filename = (os.path.basename(name))

	image = cv2.imread(name)
	orig = image.copy()

	image = cv2.resize(image, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	model = load_model(args["model"])

	(badlen, goodlen) = model.predict(image)[0]

	label = "GOOD" if goodlen> badlen else "BAD"
	proba = goodlen if goodlen > badlen else badlen
	label = "{}: {:.2f}%".format(label, proba * 100)

	output = imutils.resize(orig, width=400)
	cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
		0.7, (0, 255, 0), 2)
	print(filename+" "+label)
	cv2.imshow("Output", output)
	cv2.waitKey(0)
	cv2.imwrite(filename,output)

cv2.destroyAllWindows()
