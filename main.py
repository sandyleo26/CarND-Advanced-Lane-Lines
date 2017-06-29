import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
from moviepy.editor import VideoFileClip

def get_mtx_dist():
    '''
    Get calibratoin matrix and distortion coefficents
    '''
    objPoints = []
    imgPoints = []
    objP = np.zeros((6*9,3), np.float32)
    objP[:,0:2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

    for imgfile in glob.glob('./camera_cal/*'):
        img = mpimg.imread(imgfile)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (6,9))
        # cv2.drawChessboardCorners(img, corners)
        if not ret:
            print('findChessboardCorners failed on {}'.format(imgfile))
            continue

        objPoints.append(objP)
        imgPoints.append(corners)

    retval, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
    ## Run once and save the cameraMatrix and distCoeffs
    # np.savez('./calibration_data.npz', mtx=mtx, dist=dist)

    return mtx, dist

def abs_sobel_thresh(gray, orient='x', sobel_kernel=9, thresh=(30, 100)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    grad_binary = np.zeros_like(gray)
    grad_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return grad_binary

def mag_thresh(gray, sobel_kernel=9, mag_thresh=(30, 100)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(gray, sobel_kernel=15, thresh=(0.7, 1.3)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, gray, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, gray, sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    dir_output = np.zeros_like(gray)
    dir_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return dir_output

def color_threshold(img, thresh=(170, 255)):
    # hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # rchannel = img[:,:,0]
    # rchannel_binary = np.zeros_like(rchannel)
    # rchannel_binary[(rchannel > thresh[0]) & (rchannel < thresh[1])] = 1
    # schannel = hls[:,:,2]
    # schannel_binary = np.zeros_like(schannel)
    # schannel_binary[(schannel > thresh[0]) & (schannel < thresh[1])] = 1
    # combined_binary = np.zeros_like(rchannel)
    # combined_binary[(rchannel > 200) & (rchannel < 255)] = 1
    lower_yellow_hsv = (20, 100, 100)
    upper_yellow_hsv = (40, 255, 255)
    lower_white_rgb = (200, 200, 200)
    upper_white_rgb = (255, 255, 255)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, lower_yellow_hsv, upper_yellow_hsv)
    white = cv2.inRange(img, lower_white_rgb, upper_white_rgb)
    combined_binary = cv2.bitwise_or(yellow, white)//255


    ## Test sobel threshold and color threshold
    # fig, axes = plt.subplots(1,2,figsize=(10,6))
    # axes[0].imshow(schannel, cmap='gray')
    # axes[1].imshow(schannel_binary, cmap='gray')
    # plt.show()
    return combined_binary

## find 4 points for perspective transform
def retrieve_points_for_warping(img):
    def onclick(event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    ax.imshow(img)
    ax.grid(True)
    plt.show()

## perspective transform
def perspective_transform(img):
    gray = img
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(gray, M, gray.shape[0:2][::-1], cv2.INTER_LINEAR)

    ## Test perspective transform
    # fig, axes = plt.subplots(1,2,figsize=(10,6))
    # axes[0].imshow(gray)
    # axes[1].imshow(warped)
    # plt.show()
    return warped

## using sliding window to find lane line
def find_lane_line_sliding_windows(img):
    gray = img
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    out_img = np.dstack((gray, gray, gray))*255

    histogram = np.sum(gray[gray.shape[0]//2:,:], axis=0)
    midpoint = histogram.shape[0]//2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = np.int(gray.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = gray.nonzero()
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
        win_y_low = gray.shape[0] - (window+1)*window_height
        win_y_high = gray.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
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
    new_fit_found = True
    if len(lefty) == 0 or len(righty) == 0:
        return False, np.array([]), np.array([])
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, gray.shape[0]-1, gray.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    ## visualize
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # #plt.xlim(0, 1280)
    # #plt.ylim(720, 0)
    # plt.show()

    return new_fit_found, left_fit, right_fit

## find lane lines using polyfit parameters returned from previous blind search (e.g. sliding windows)
def find_lane_line(img, left_fit, right_fit):
    gray = img
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    nonzero = gray.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    new_fit_found = True
    if len(leftx) == 0 or len(rightx) == 0:
        new_fit_found = False
        new_left_fit = left_fit
        new_right_fit = right_fit
    else:
        # Fit a second order polynomial to each
        new_left_fit = np.polyfit(lefty, leftx, 2)
        new_right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, gray.shape[0]-1, gray.shape[0] )
    left_fitx = new_left_fit[0]*ploty**2 + new_left_fit[1]*ploty + new_left_fit[2]
    right_fitx = new_right_fit[0]*ploty**2 + new_right_fit[1]*ploty + new_right_fit[2]
    vehicley = ploty[-1]
    vehicelx = (left_fitx[-1] + right_fitx[-1]) // 2

    ## Create an image to draw on and an image to show the selection window
    out_img = np.dstack((gray, gray, gray))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    # plt.imshow(result)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()
    return new_fit_found, new_left_fit, new_right_fit, left_fitx, right_fitx, ploty

## calculate radius and center offset
def curvature_and_offset(img, leftx, rightx, ploty):
    if len(leftx) == 0 or len(rightx) == 0:
        return -1, -1, -1
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/590 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    lane_center_offset = (rightx[-1] + leftx[-1] - img.shape[1]//2)//2 * xm_per_pix
    return left_curverad, right_curverad, lane_center_offset

## unwarp birdeye view to normal view
def unwarp(undist, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(undist[:,:,2]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    newwarp = color_warp
    if len(left_fitx) != 0 and len(right_fitx) != 0:
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        Minverse = cv2.getPerspectiveTransform(dst, src)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minverse, (undist.shape[1], undist.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # plt.imshow(result)
    # plt.show()
    return result, color_warp

def process_image(img):
    undist = cv2.undistort(img, mtx, dist)
    gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
    gradx = abs_sobel_thresh(gray)
    mag_binary = mag_thresh(gray)
    dir_binary = dir_threshold(gray)
    sobel_combined = dir_binary & mag_binary
    
    color_binary = color_threshold(undist)
    combined = color_binary | sobel_combined
    warped = perspective_transform(combined)
    original_warped = perspective_transform(undist)

    left_fitx, right_fitx, ploty = np.array([]), np.array([]), np.linspace(0, gray.shape[0]-1, gray.shape[0] )
    if line.detected:
        new_fit_found, new_left_fit, new_right_fit, left_fitx, right_fitx, _ = find_lane_line(
                warped, line.current_fit[0], line.current_fit[1])
        if not new_fit_found:
            line.debugOut = True
            line.detected = False
            print('find_lane_line found cannot polyfit. Check debug/{}.jpg'.format(line.debugCount))
        else:
            line.current_fit = [new_left_fit, new_right_fit]
    else:
        new_fit_found, left_fit, right_fit = find_lane_line_sliding_windows(warped)
        if new_fit_found:
            new_fit_found, new_left_fit, new_right_fit, left_fitx, right_fitx, _ = find_lane_line(
                    warped, left_fit, right_fit)
            assert(new_fit_found)
            line.detected = True
            line.current_fit = [new_left_fit, new_right_fit]
        else:
            line.debugOut = True
            print('find_lane_line_sliding_windows cannot polyfit. Check debug/{}.jpg'.format(line.debugCount))

    unwarped, color_warp = unwarp(undist, left_fitx, right_fitx, ploty)
    left_radius, right_radius, center_offset = curvature_and_offset(unwarped, left_fitx, right_fitx, ploty)

    # sanity check left&right radius
    absolute_diff = abs(left_radius-right_radius)
    if enableDebug and (absolute_diff > .2*abs(left_radius) or absolute_diff > .2*abs(right_radius)):
        line.debugOut = True
        print('Sanity check failed for left and right radius. Use {}.jpg to debug'.format(line.debugCount))
        print(left_radius, 'm', right_radius, 'm', center_offset, 'm')

    cv2.putText(unwarped, 'Radius of Curvature = {}(m)'.format((left_radius+right_radius)//2), (50,100),
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(unwarped, 'Vehicle is {:.2f}m off center'.format(center_offset), (50,160),
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
    if enableDebug and line.debugOut:
        line.debugOut = False
        fig, axes = plt.subplots(3,4, figsize=(24,15))
        fig.tight_layout()
        images = [img, unwarped, gray, gradx, mag_binary, dir_binary, sobel_combined, color_binary, combined, warped, color_warp, original_warped]
        titles = ['img', 'unwarped', 'gray', 'gradx', 'mag_binary', 'dir_binary', 'sobel_combined', 'color_binary', 'combined', 'warped', 'color_warp', 'original_warped']
        for i, image in enumerate(images):
            if i == 0 or i == 1:
                fig.axes[i].imshow(image)
            else:
                fig.axes[i].imshow(image, cmap='gray')
            fig.axes[i].set_title(titles[i])
        # axes[0,0].imshow(img)
        # axes[0,0].set_title('Original')
        # axes[0,1].imshow(unwarped)
        # axes[0,1].set_title('Unwarped')
        # axes[0,2].imshow(sobel_binary)
        # axes[0,2].set_title('Sobel Binary')
        # axes[1,0].imshow(color_binary)
        # axes[1,0].set_title('Color Binary')
        # axes[1,1].imshow(binary_img)
        # axes[1,1].set_title('Combined Binary')
        # axes[1,2].imshow(warped)
        # axes[1,2].set_title('Warped')
        plt.savefig('debug/{}.jpg'.format(line.debugCount))
        cv2.imwrite('debug/original{}.jpg'.format(line.debugCount), img)
        line.debugCount += 1

    return unwarped

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        # debug image count
        self.debugOut = False
        self.debugCount = 0

# manually selected points for warping. Make sure straightline look straight
src = np.float32([[(200, 720), (560, 470), (720, 470), (1130, 720)]])
dst = np.float32([[(320, 720), (320, 0), (920, 0), (920, 720)]])
# load calibration data
calibration_data = np.load('./calibration_data.npz')
mtx, dist = calibration_data['mtx'], calibration_data['dist']
# Test images
calibration_img = mpimg.imread('./camera_cal/calibration1.jpg')
img = mpimg.imread('./test_images/straight_lines1.jpg')
# img = mpimg.imread('./debug/original2.jpg')

## process video
enableDebug = True
# video_in = 'project_video.mp4'
video_in = 'challenge_video.mp4'
# video_in = 'harder_challenge_video.mp4'
video_out = video_in.split('.')[0] + '_processed.' + video_in.split('.')[1]
clip = VideoFileClip(video_in)
line = Line()
clip_processed = clip.fl_image(process_image)
clip_processed.write_videofile(video_out, audio=False)
# retrieve_points_for_warping(img)
# process_image(img)
