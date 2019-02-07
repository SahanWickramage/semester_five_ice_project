import numpy as np
import cv2

def select_rgb_white_yellow(image): 
    # white color mask
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

img = cv2.imread('road.jpg')
cv2.imshow('input', img)
img_out = select_rgb_white_yellow(img)
cv2.imshow('output',img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
