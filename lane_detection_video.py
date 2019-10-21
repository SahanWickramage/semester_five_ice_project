#import libraries
import numpy as np
import cv2
import glob
from io import BytesIO
from IPython.display import Image
from tqdm import tqdm
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed

#define functions
# Helper functions for displaying images within the notebook




def show_RGB(img1, img2, title1='Undistorted', title2='Binary', chan1='BGR', chan2='GRAY'):
    '''Displays two images side-by-side using Matplotlib'''
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.set_axis_off(), ax2.set_axis_off()
    
    if chan1 == 'BGR':
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    if chan2 == 'BGR':
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)    
        
    ax1.imshow(img1, cmap='gray')
    ax1.set_title(title1, fontsize=20)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(title2, fontsize=20)

# Undistort function
def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

# Directional gradient for X or Y using Sobel algorithm

def abs_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Absolulte value of x or y gradient 
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    
    # Rescale to 8-bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    
    # Apply threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Gradient magnitude using both X and Y

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # X and Y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Gradient magnitude
    mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Rescale to 8-bit integer
    scaled_sobel = np.uint8(255 * mag / np.max(mag))
    
    # Apply threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# CLAHE threshold function

def clahe_thresh(img, thresh=182):
    '''Applies the Contrast Limited Adaptive Histogram Equalization (CLAHE) algorithm.  
    '''
    
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    
    # Apply smoothing
    img = cv2.medianBlur(img, 3)
    
    # Apply CLAHE
    img = cv2.equalizeHist(img)
    
    # Apply threshold and create binary
    ret, clahe_binary = cv2.threshold(img, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)

    return clahe_binary

def hls_thresh(img, channel='S', thresh=(200, 255)):
    
    # Convert to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    # Split the image into its color channels
    H, L, S = hls[:,:,0], hls[:,:,1], hls[:,:,2]
    
    # Select the channel to use
    if channel == 'H':
        channel = H
    elif channel == 'L':
        channel = L
    elif channel == 'S':
        channel = S
    else:
        raise ValueError("Channel not recognized. Only HLS channels can be displayed as binaries.")
    
    # Create binary mask using the chosen channel
    hls_binary = np.zeros_like(channel)
    hls_binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    
    return hls_binary

def hsv_thresh(img, channel='S', thresh=(200, 255)):
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Split the image into its color channels
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    # Select the channel to use
    if channel == 'H':
        channel = H
    elif channel == 'S':
        channel = S
    elif channel == 'V':
        channel = V
    else:
        raise ValueError("Channel not recognized. Only HSV channels can be displayed as binaries.")
    
    # Create binary mask using the chosen channel
    hsv_binary = np.zeros_like(channel)
    hsv_binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    
    return hsv_binary

# Combined thresholding function
def comb_thresh(img):
        
    # Gradient thresholds
    sx_binary = abs_thresh(img, orient='x', sobel_kernel=3, thresh=(30,130))
    sy_binary = abs_thresh(img, orient='y', sobel_kernel=3, thresh=(50,140))
    mag_binary = mag_thresh(img, sobel_kernel=5, thresh=(50,150))
    
    # Color thresholds
    clahe_binary = clahe_thresh(img)  # contrast limited adaptive hist. equal.
    s_binary = hls_thresh(img, channel='S', thresh=(120,255))  # saturation channel
    v_binary = hsv_thresh(img, channel='V', thresh=(225,255))  # value channel

    ## Combinations of above thresholds:
    
    # Gradients
    grad = np.zeros_like(sx_binary)
    grad[((sx_binary == 1) | (sy_binary == 1)) | (mag_binary == 1)] = 1
    
    # S + V channels
    sv = ((s_binary == 1) | (v_binary == 1))
    
    # CLAHE + SV
    csv = ((clahe_binary == 1) | (s_binary == 1) | (v_binary == 1))
    
    # Gradient + SV
    gradsv = ((grad == 1) | (sv == 1))

    # Return various combinations to test
    return sv, gradsv

# Perspective transform functions
def get_vertices(img):

    # Get image dimensions
    img_size = (img.shape[1], img.shape[0])
    img_width, img_height = img.shape[1], img.shape[0]
    
    # Define the region of interest
    y_mid = img_size[0]/2   # midpoint of image width y
    x_mid = img_size[1]/2   # midpoint of image height x
    y_up_off = 80           # y offset from horizontal midpoint for calculating upper vertices of ROI polynomial
    y_low_off = 450         # y offset from horizontal midpoint for calculating lower vertices of ROI polynomial
    x_up_off = 110          # x offset from vertical midpoint for calculating upper vertices of ROI polynomial
    x_low_off = 350         # x offset from vertical midpoint for calculating lower vertices of ROI polynomial
    
    points = [
        (y_mid - y_up_off, x_mid + x_up_off),   
        (y_mid + y_up_off, x_mid + x_up_off),
        (y_mid + y_low_off, x_mid + x_low_off),
        (y_mid - y_low_off, x_mid + x_low_off),
    ]
    src = np.float32(points)
    
    # Define warp points as dst 
    dst = np.float32([
        [y_mid - y_low_off, 0],
        [y_mid + y_low_off, 0],
        [y_mid + y_low_off, img_height],
        [y_mid - y_low_off, img_height],
    ])
    
    return src, dst


def perspective_transform(img, src, dst):

    # Calculate perspective transforms
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Apply perspective transformation to image
    img_size = (img.shape[1], img.shape[0])       
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, M, Minv


def draw_lines(img, vertices):
    points= np.int32([vertices])
    img_lines = cv2.polylines(img, points, True, (0,255,0), thickness=2)
    
    return img_lines

# Use warped binary to identify lane lines using sliding windows approach
def find_lines(binary_warped, show=False):

    # Make sure binary doesn't have float values
    binary_warped = binary_warped.astype('uint8')
    
    if show:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 100
    
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if show:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)         
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if show:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        fig2 = plt.figure(figsize = (10,10)) # create a 5 x 5 figure 
        ax3 = fig2.add_subplot(1, 1, 1)
        ax3.imshow(out_img, interpolation='none')
        #ax3.plot(left_fitx, ploty, color='yellow')
        #ax3.plot(right_fitx, ploty, color='yellow')
        ax3.plot(left_fit, ploty, color='yellow') #I changed left_fitx to leftx on 20th March 2019 9.02 p.m.
        ax3.plot(right_fit, ploty, color='yellow') #I changed right_fitx to rightx on 20th March 2019 9.02 p.m.
        ax3.set_title('Sliding Windows & Polynomial Fit')
        plt.show()  
        
    return left_fit, right_fit

# Function that uses the last frame as reference for fitting the next frame
def use_last_frame(binary_warped, left_fit, right_fit):

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 30
    left_lane_inds = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
        (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
        (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
        
    return left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx, lefty, righty, left_lane_inds, right_lane_inds

def radius_center(ploty, leftx, rightx, lefty, righty):
    
    y_eval = np.max(ploty)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curve_radius = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curve_radius = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    radius = np.mean([left_curve_radius, right_curve_radius])

    left_y_max = np.argmax(lefty)
    right_y_max = np.argmax(righty)
    center_x = (leftx[left_y_max] + rightx[right_y_max])/2
    center_offset = (640 - center_x) * xm_per_pix
    
    
    return radius, center_offset

# Function for drawing complete lane markings back onto the original image
def project_back(binary_warped, original_image, undist, Minv, left_fitx, right_fitx, ploty):
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv,
                                  (original_image.shape[1], original_image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result

# Moving average function
def moving_average(radius, center_offset, left, right, plot):

    global count
    if count == 5:
        recent_radius.pop(0)
        recent_center_offset.pop(0)
        recent_left_fitx.pop(0)
        recent_right_fitx.pop(0)
        recent_ploty.pop(0)
    else:
        count += 1
    recent_radius.append(radius)
    recent_center_offset.append(center_offset)
    recent_left_fitx.append(left)
    recent_right_fitx.append(right)
    recent_ploty.append(plot)
    if count > 5:
        exit()

def pipeline(image):

    undist = undistort(image)
    src, dst = get_vertices(undist)
    warped, M, Minv = perspective_transform(undist, src, dst)
    warped_binary_sv, warped_binary = comb_thresh(warped)
    
    global first, left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx, lefty, righty
    
    if first:
        left_fit, right_fit = find_lines(warped_binary)
        left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx, lefty, righty, _, _ = \
            use_last_frame(warped_binary, left_fit, right_fit)
        first = False

    left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx, lefty, righty, _, _ = \
        use_last_frame(warped_binary, left_fit, right_fit)
    
    radius, center_offset = radius_center(ploty, leftx, rightx, lefty, righty)

    moving_average(radius, center_offset, left_fitx, right_fitx, ploty)

    result = project_back(warped_binary, image, undist, Minv,
                          np.add.reduce(recent_left_fitx) / count,
                          np.add.reduce(recent_right_fitx) / count,
                          np.add.reduce(recent_ploty) / count)

    cv2.putText(result, 'Radius: {0:.1f}m '.format(np.add.reduce(recent_radius) / count), (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, bottomLeftOrigin=False)
    cv2.putText(result, 'Center Offset: {0:.2f}m'.format(np.add.reduce(recent_center_offset) / count), (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, bottomLeftOrigin=False)
    
    
    return result
    #return warped_binary

# Set global variables for moving average
count = 0
first = True
left_fitx = None
right_fitx = None
ploty = None
left_fit = None
right_fit = None
leftx = None
rightx = None
lefty = None
righty = None

# Create placeholder lists for storing recent values
recent_radius = []
recent_center_offset = []
recent_left_fitx = []
recent_right_fitx = []
recent_ploty = []

#calibrate the camera
# Object point and image point placeholders
nx = 9
ny = 6
objp = np.zeros((ny*nx, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# List of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Locate the chessboard corners
for fname in tqdm(images):
    # Get image array and convert to grayscale
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

print('Calibration complete.')

## Test the calibration by undistorting a chessboard image
img_dir = 'camera_cal/'

# Test image
test_img = 'calibration1.jpg'
img = cv2.imread(img_dir + test_img)
img_size = (img.shape[1], img.shape[0])

# Camera calibration 
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)


#call defined functions to get desired output
#test the code
#image3 = cv2.imread('test_images/test3.jpg')
#test3_marked = pipeline(image3)
#cv2.imshow('output',test3_marked)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

#VIDEO_NAME = 'vehicle_detection_raw_video.mp4'
#VIDEO_NAME = 'Entering and exiting the Southern Expressway in Sri Lanka through the new Kottawa Interchange.mp4'
#VIDEO_NAME = 'Driving in Delhi (Delhi-Meerut Expressway) - India.mp4'
VIDEO_NAME = 'day.avi'
#VIDEO_NAME = 'MOVA1760.mp4'

video = cv2.VideoCapture(VIDEO_NAME)

while(video.isOpened()):
    ret, frame = video.read()

    # All the results have been drawn on the frame, so it's time to display it.
    try:
        frame_marked = pipeline(frame)
        cv2.imshow('output', frame_marked)
    except:
        pass
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
