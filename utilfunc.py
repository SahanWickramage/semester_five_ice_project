import numpy as np
import cv2

#we not going to calibrate the camera, we already have camera marix and distortion cooefficients
mtx_temp = [[1.15777930e+03, 0.00000000e+00, 6.67111054e+02], [0.00000000e+00, 1.15282291e+03, 3.86128937e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
dist_temp = [[-0.24688775, -0.02373133, -0.00109842,  0.00035108, -0.00258571]]

mtx = np.ndarray(shape=(3,3), dtype=float)
for row in range(3):
    for column in range(3):
        mtx[row][column] = mtx_temp[row][column]
del(mtx_temp)

dist = np.ndarray(shape=(1,5), dtype=float)
for row in range(1):
    for column in range(5):
        dist[row][column] = dist_temp[row][column]
del(dist_temp)

print(mtx)
print(dist)

# Undistort function
def undistort(img):
    global mtx, dist
    return cv2.undistort(img, mtx, dist, None, mtx)

# Perspective transform functions (two functions)
#function 1
def get_vertices(img, condition):

    # Get image dimensions
    img_width, img_height = img.shape[1], img.shape[0]
 
    # define the region of interest
    y_mid = img_height//2  # midpoint of image height y
    x_mid = img_width//2   # midpoint of image width x

    if condition == 'n':
        #parameters for night
        y_up_off = 0
        y_down_off = 290
        x_up_left_off = 40
        x_up_right_off = 70
        x_down_left_off = 350
        x_down_right_off = 400
    
    elif condition == 'd':
        #parameters for day
        y_up_off = -100
        y_down_off = 310
        x_up_left_off = 80
        x_up_right_off = 90
        x_down_left_off = 370
        x_down_right_off = 490

    points = [
        (x_mid - x_up_left_off, y_mid - y_up_off),   
        (x_mid + x_up_right_off, y_mid - y_up_off),
        (x_mid + x_down_right_off, y_mid + y_down_off),
        (x_mid - x_down_left_off, y_mid + y_down_off)
    ]
    
    src = np.float32(points)

    dst = np.float32([
        [0, 0],
        [img_width, 0],
        [img_width, img_height],
        [0, img_height]
    ])
    
    return src, dst

#function 2
def perspective_transform(img, src, dst):

    # Calculate perspective transforms
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Apply perspective transformation to image
    img_size = (img.shape[1], img.shape[0])       
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, M, Minv

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

def comb_thresh(image, condition):

    Xsobel_kernel=3
    Xthresh_min=30
    Xthresh_max=130
    Ysobel_kernel=3
    Ythresh_min=50
    Ythresh_max=140
    mag_sobel_kernel=5
    mag_thresh_min=50
    mag_thresh_max=150
    Sthresh_min=225
    Sthresh_max=255

    if condition == 'n':
        Vthresh_min=120
    
    elif condition == 'd':
        Vthresh_min=225
    
    Vthresh_max=255
 
    # Gradient thresholds
    sx_binary = abs_thresh(image, orient='x', sobel_kernel=Xsobel_kernel, thresh=(Xthresh_min,Xthresh_max))
    sy_binary = abs_thresh(image, orient='y', sobel_kernel=Ysobel_kernel, thresh=(Ythresh_min,Ythresh_max))
    mag_binary = mag_thresh(image, sobel_kernel=mag_sobel_kernel, thresh=(mag_thresh_min,mag_thresh_max))
    
    # Color thresholds
    s_binary = hsv_thresh(image, channel='S', thresh=(Sthresh_min,Sthresh_max))  # saturation channel
    v_binary = hsv_thresh(image, channel='V', thresh=(Vthresh_min, Vthresh_max))  # value channel

    ## Combinations of above thresholds:
    
    # Gradients
    grad = np.zeros_like(sx_binary)
    grad[((sx_binary == 1) | (sy_binary == 1)) | (mag_binary == 1)] = 1
    
    # S + V channels
    sv = np.zeros_like(s_binary)
    sv[(s_binary == 1) | (v_binary == 1)] = 1
    
    # Gradient + SV
    gradsv = np.zeros_like(grad)
    gradsv[(grad == 1) | (sv == 1)] = 1
    return gradsv

# Use warped binary to identify lane lines using sliding windows approach
def find_lines(binary_warped):
    
    binary_warped = binary_warped.astype('uint8')

    #originate the output image
    image_out = np.dstack((binary_warped, binary_warped, binary_warped))*255

    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9

    window_height = np.int(binary_warped.shape[0]/nwindows)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100

    minpix = 50

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        #draw the sliding window in the output image
        cv2.rectangle(image_out,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(image_out,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

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

    return left_fit, right_fit, image_out, histogram

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

def pipeline(image, condition):

    undist = undistort(image)
    src, dst = get_vertices(undist, condition)
    warped, M, Minv = perspective_transform(undist, src, dst)
    warped_binary = comb_thresh(warped, condition)
    
    #global left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx, lefty, righty
    
    left_fit, right_fit, perspective, histogram = find_lines(warped_binary)
    left_fitx, right_fitx, ploty, left_fit, right_fit, leftx, rightx, lefty, righty, _, _ = \
    use_last_frame(warped_binary, left_fit, right_fit)

    radius, center_offset = radius_center(ploty, leftx, rightx, lefty, righty)

    result = project_back(warped_binary, image, undist, Minv, left_fitx, right_fitx, ploty)

    cv2.putText(result, 'Radius: {0:.1f}m '.format(radius), (50, 50), 
    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, bottomLeftOrigin=False)
    cv2.putText(result, 'Center Offset: {0:.2f}m'.format(center_offset), (50, 100),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, bottomLeftOrigin=False)
    if condition == 'n':
        text = 'night'
    elif conditin == 'd':
        text = 'day'
    cv2.putText(result, 'Condition: '+text, (50, 150),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, bottomLeftOrigin=False)
    
    return result, perspective, histogram