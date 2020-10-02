
import tensorflow
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import glob
import os



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
