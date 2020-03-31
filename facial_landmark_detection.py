# Face landmarks Detection
# usage:
# python facelandmarkdetect.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/face1.jpg

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import os
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
import face_recognition

font = cv2.FONT_HERSHEY_SIMPLEX

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

if os.path.isfile(args["shape_predictor"]):
	pass
else:
	# print("Oops...! File is not available. Shall I downlaod ?")
	cmd = "wget -c --progress=bar http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
	os.system(cmd)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = face_recognition.load_image_file(args["image"])
orig = image
image = imutils.resize(image, width=500)

print(image.shape)
print(image.dtype)
#gray = np.array(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray.shape)
print(gray.dtype)
# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the face number
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	s = [shape[17],shape[26],shape[0],shape[3],shape[5],shape[11],shape[13],shape[16]]
	

	count = 0
	avg_g = []
	for (x, y) in shape:	
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
		count+=1
		G = []
		print(x,y)
		for i in range(-1,2):
			for j in range(-1,2):
				G.append(gray[y-j][x-i])
				
		avg_g.append(sum(G)/9)			

	print(len(G)) 
	print(len(avg_g))
	for number, (x, y) in enumerate(shape):
		r =1
		text = str(number)
		(tw, th), bl = cv2.getTextSize(text, font, 0.5, 2)      # So the text can be centred in the circle
		tw /= 2
		th = th / 2 + 2

		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.rectangle(image, 
					 (int(x - tw), int(y - th)), 
					 (int(x + tw), int(y + th)), 
					 (0, 128, 255), 
					 -1)
		# number each circle, centred in the rectangle
		cv2.putText(image, text, (int(x-tw), int(y + bl)), font, 0.5, (0,0,0), 2)
		# get the average value in specified sample size (20 x 20)







# show the output image with the face detections + facial landmarks
plt.subplot(121)
plt.imshow(orig)
plt.xticks([])
plt.yticks([])
plt.title("Intput")

plt.subplot(122)
plt.imshow(image)
plt.xticks([])
plt.yticks([])
plt.title("Output")

fname = "result_" + args["image"][1]
#plt.savefig(fname)
plt.show()