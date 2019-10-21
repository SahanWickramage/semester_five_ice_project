import numpy as np
import cv2

def color_thresh_combined(img, s_thresh, l_thresh, v_thresh, b_thresh):
    V_binary = HSV_thresh(img, v_thresh)
    S_binary = HLS_thresh(img, s_thresh)
    L_binary = LUV_thresh(img, l_thresh)
    color_binary = np.zeros_like(V_binary)                           
    color_binary[(V_binary == 1) & (S_binary == 1) & (L_binary == 1)] = 1
return color_binary

img = cv2.imread('road.jpg')

img_hsv = 
