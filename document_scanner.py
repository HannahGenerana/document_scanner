# pseudocode
import cv2
import numpy as np 
from imutils.perspective import four_point_transform
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# functions
def scan_detection(image):
    global document_contour

    document_contour = np.array([[0,0], [width, 0], [width, height], [0, height]])
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted (contours, key = cv2.contourArea, reverse=True)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area


# video set up
cap = cv2.VideoCapture (0 + cv2.CAP_DSHOW)
width, height = 500, 450
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


while True:
    _, frame = cap.read()
    frame_copy = frame.copy()
# detect the paper to scan
    scan_detection (frame_copy)
    cv2.imshow("input", frame)

# scan wrapping
    warped = four_point_transform(frame_copy, document_contour.reshape(4,2))
    cv2.imshow ("warped", warped)

# convert image to black and white

# create an optical characted recognition
    ocr_text = pytesseract.image_to_string(warped)
    print (ocr_text)
# save the image and text to be printed

#https://www.youtube.com/watch?v=W3DzSm8WI1g - yt link of the code 