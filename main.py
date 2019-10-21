import numpy as np
import cv2
import glob
#import matplotlib.pyplot as plt

from utilfunc import *

condition = 'n'
VIDEO_NAME = 'night.mp4'

video = cv2.VideoCapture(VIDEO_NAME)

while(video.isOpened()):
    ret, frame = video.read()

    # All the results have been drawn on the frame, so it's time to display it.
    try:
        result, perspective, _ = pipeline(frame, condition)
        s_frame = cv2.resize(frame, (640, 360))
        cv2.imshow('input',s_frame)
        s_result = cv2.resize(result, (640, 360))
        cv2.imshow('result',s_result)
        S_perspective = cv2.resize(perspective, (640, 360)) 
        cv2.imshow('perspective',S_perspective)
    except:
        pass
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()