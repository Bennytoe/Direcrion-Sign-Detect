import cv2
import numpy as np

def is_circle(contour):
    
    epsilon = 0.02 * cv2.arcLength(contour, True) 
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # proverka za krugla forma
    return len(approx) >= 8

def detect_and_draw_sign(image):
    # select image 
    
    image = cv2.imread("g1.jpg")
    
    
    image = cv2.resize(image,(500,600)) # resizng
    
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # converting to hsv

    # Opredelqne na oblastite na cvetovete koito otgovarqt za cvetovete na putniq znak
    
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # namirane na kontura na znaka

    # Ograjdane na znaka
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 5000 and is_circle(contour):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # pokazvane na rezultat
    cv2.imshow('Detect Sign', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_and_draw_sign('recog_image.jpg')