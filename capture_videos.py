import cv2
import numpy as np

VIDEO_NAME = 'vehicle_detection_raw_video.mp4'
#video = cv2.VideoCapture(PATH_TO_VIDEO)
video = cv2.VideoCapture(VIDEO_NAME)

while(video.isOpened()):
    ret, frame = video.read()

    # All the results have been drawn on the frame, so it's time to display it.
    try:
        cv2.imshow('input_video', frame)
    except:
        pass

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
