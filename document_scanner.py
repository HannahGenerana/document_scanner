import cv2
import numpy as np 
from imutils.perspective import four_point_transform
import pytesseract

# pseudocode

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# variables
font = cv2.FONT_HERSHEY_SIMPLEX
count = 0
scale = 1

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


def image_processing (image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    return threshold


def center_text(image, text):
    text_size = cv2.getTextSize(text, font, 2, 5)[0]
    first_text = (image.shape[1] - text_size[0]) // 2
    second_text = (image.shape[0] - text_size[1]) // 2
    cv2.putText(image, text, (first_text, second_text), font, 2, (255,0), 5, cv2.LINE_AA)



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
    cv2.imshow("Input", cv2.resize(frame, (int(scale * width), int(scale * height))))

# scan wrapping
    warped = four_point_transform(frame_copy, document_contour.reshape(4,2))
    cv2.imshow ("Warped", cv2.resize(warped, (int(scale * warped.shape[1]), int(scale * warped.shape[0]))))

# convert image to black and white
    processed = image_processing(warped)
    processed = processed[10:processed.shape[0]-10, 10:processed.shape[1]-10]
    cv2.imshow("Processed", processed)

# create an optical characted recognition
    ocr_text = pytesseract.image_to_string(warped)

# save the image and text to be printed
    pressed_keys = cv2.waitKey(1) & 0xFF
    if pressed_keys == 27:
        break

    elif pressed_keys == ord('s'):
        cv2.imwrite('Outpute/scanned_' + str(count) + ".jpg", processed)
        count += 1

        center_text (frame, "Scan saved")
        cv2.imshow("Input", cv2.resize(frame, (int(scale * width), int(scale * height))))
        cv2.waitKey(500)

    elif pressed_keys == ord('o'):
        file = open("output/recognized_" + str(count - 1) + ".txt", "w")
        ocr_text = pytesseract.image_to_string(warped)
        file.write(ocr_text)
        file.close()

        center_text (frame, "scan saved")
        cv2.imshow("Input", cv2.resize(frame, (int(scale * width), int(scale * height))))
        cv2.waitKey(500)

cv2.destroyAllWindows()

#https://www.youtube.com/watch?v=W3DzSm8WI1g - yt link of the code 