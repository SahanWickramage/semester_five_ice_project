import numpy as np
import cv2

img = cv2.imread('road.jpg')
cv2.imshow('input', img)

#binarization
lower = np.uint8([230, 230, 230])
upper = np.uint8([255, 255, 255])

img_out_binarization = cv2.inRange(img, lower, upper)
cv2.imshow('binarization',img_out_binarization)

#bluring
img_out_blur = cv2.blur(img_out_binarization,(5,5))
cv2.imshow('blur',img_out_blur)

#canny edge detection
img_out_canny_edge_detection = cv2.Canny(img_out_blur,100,200)
cv2.imshow('canny edge detection',img_out_canny_edge_detection)

#region of interest
# first, define the polygon by vertices
rows, cols = img_out_canny_edge_detection.shape[:2]
bottom_left  = [cols*0.2, rows*1]
top_left     = [cols*0.4, rows*0.4]
bottom_right = [cols*0.95, rows*1]
top_right    = [cols*0.8, rows*0.4] 
#the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

#filter out the required region of interest
mask_roi = np.zeros_like(img_out_canny_edge_detection)
if len(mask_roi.shape)==2:
    cv2.fillPoly(mask_roi, vertices, 255)
else:
    cv2.fillPoly(mask_roi, vertices, (255,)*mask_roi.shape[2]) # in case, the input image has a channel dimension        

img_roi = cv2.bitwise_and(img_out_canny_edge_detection, mask_roi)
cv2.imshow('region of interest',img_roi)

#probabilistic hough transformation
lines_hough_transformation = cv2.HoughLinesP(img_roi, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
print(lines_hough_transformation)

for line in lines_hough_transformation:
    coordinates = line[0]
    print(coordinates[0]+coordinates[1]+coordinates[2]+coordinates[3])
    cv2.line(img,(coordinates[0],coordinates[1]),(coordinates[2],coordinates[3]),[0,0,255],3)
    
cv2.imshow('hough transformation',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
