# pseudocode
import cv2
import numpy as np 
from imutils.perspective import four_point_transform
import pytesseract

# video set up
cap = cv2.VideoCapture (0 + cv2.CAP_DSHOW)
width, height = 500, 450
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# detect the paper to scan

# scan wrapping

# convert image to black and white

# create an optical characted recognition

# save the image and text to be printed

#https://www.youtube.com/watch?v=W3DzSm8WI1g - yt link of the code 