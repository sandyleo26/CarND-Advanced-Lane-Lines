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

def abs_threshold(gray, orient='x', sobel_kernel=9, thresh=(30, 100)):
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        
    abs_sobel = np.absolute(sobel) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(gray)
    grad_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return grad_binary

def mag_threshold(gray, sobel_kernel=9, mag_thresh=(30, 100)):
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

def hsv_threshold(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_yellow_hsv = (20, 50, 100)
    upper_yellow_hsv = (40, 255, 255)
    yellow = cv2.inRange(hsv, lower_yellow_hsv, upper_yellow_hsv)
    return yellow & np.ones_like(yellow)

def rgb_threshold(img):
    lower_white_rgb = (200, 200, 200)
    upper_white_rgb = (255, 255, 255)
    white = cv2.inRange(img, lower_white_rgb, upper_white_rgb)
    return white & np.ones_like(white)

def hls_threshold(img, thresh=(90, 200)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    schannel = hls[:,:,2]
    schannel_binary = np.zeros_like(schannel)
    schannel_binary[(schannel > thresh[0]) & (schannel < thresh[1])] = 1
    return schannel_binary

def color_threshold(img):
    hsv_binary = hsv_threshold(img)
    hls_binary = hls_threshold(img)
    rgb_binary = rgb_threshold(img)
    result = hsv_binary | hls_binary | rgb_binary
    return result

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
def sliding_window_find(original, gray):
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
    margin = tracker.window_margin
    # Set minimum number of pixels found to recenter window
    minpix = tracker.minpix
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
    if len(lefty) == 0 or len(righty) == 0:
        tracker.log('sliding_window_find cannot polyfit.', original, out_img)
        return tracker.get_average_fitx()

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if not tracker.checkSanity(left_fit, right_fit):
        ploty = np.linspace(0, img_height-1, img_height)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        out_img[left_fitx, ploty] = [255,255,0] # yellow
        out_img[right_fitx, ploty] = [255,255,0] # yellow
        tracker.log('sliding_window_find sanity check failed.', original, out_img)

    ## visualize
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # #plt.xlim(0, 1280)
    # #plt.ylim(720, 0)
    # plt.show()

    return tracker.get_average_fitx()

## find lane lines using polyfit parameters returned from previous blind search (e.g. sliding windows)
def quick_find(original, gray):
    ploty = np.linspace(0, img_height-1, img_height).astype(np.int)
    nonzero = gray.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = tracker.window_margin
    left_fit, right_fit = tracker.get_average_fit()
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    if len(leftx) == 0 or len(rightx) == 0:
        out_img = np.dstack((gray, gray, gray))*255
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        out_img[ploty, tracker.get_average_fitx()[0]] = [255,255,0] # yellow
        out_img[ploty, tracker.get_average_fitx()[1]] = [255,255,0] # yellow

        window_img = np.zeros_like(out_img)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        ## Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        tracker.log('quick_find cannot polyfit.', original, result)
        return tracker.get_average_fitx()

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if not tracker.checkSanity(left_fit, right_fit):
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        out_img[left_fitx, ploty] = [255,255,0] # yellow
        out_img[right_fitx, ploty] = [255,255,0] # yellow
        tracker.log('quick_find sanity check failed.', original, out_img)

    ### Create an image to draw on and an image to show the selection window
    # window_img = np.zeros_like(out_img)
    ### Color in left and right line pixels
    # left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    # left_line_pts = np.hstack((left_line_window1, left_line_window2))
    # right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    # right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    # result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    # plt.imshow(result)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()
    return tracker.get_average_fitx()

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

    lane_center_offset = ((rightx[-1] + leftx[-1])/2 - img.shape[1]/2) * xm_per_pix
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
    color_binary = color_threshold(undist)
    combined = color_binary
    combined_warped = perspective_transform(combined)
    original_warped = perspective_transform(undist)

    left_fitx, right_fitx, ploty = np.array([]), np.array([]), np.linspace(0, img_height-1, img_height)
    if tracker.get_average_fitx() != None:
        left_fitx, right_fitx = quick_find(img, combined_warped)
    else:
        left_fitx, right_fitx = sliding_window_find(img, combined_warped)

    unwarped, color_warp = unwarp(undist, left_fitx, right_fitx, ploty)
    left_radius, right_radius, center_offset = curvature_and_offset(unwarped, left_fitx, right_fitx, ploty)

    cv2.putText(unwarped, 'Radius of Curvature = {}(m)'.format((left_radius+right_radius)//2), (50,100),
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(unwarped, 'Vehicle is {:.2f}m off center'.format(center_offset), (50,160),
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

    return unwarped

# Define a class to receive the characteristics of each line detection
class Tracker():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.left_fitx_history = np.array([])
        self.right_fitx_history = np.array([])
        #average x values of the fitted line over the last n iterations
        self.average_fitx = None
        self.average_fit = None
        # debug image count
        self.debugOut = False
        self.enableDebug = False
        self.debugDir = ''
        self.debugCount = 0
        self.avgFactor = 10
        self.diffMargin = 50
        self.ploty = np.linspace(0, img_height-1, img_height)
        self.window_margin = 50
        self.minpix = 50

    def reset(self):
        self.__init__()

    def recalculate_average(self):
        self.average_fitx = (
                np.clip(np.average(self.left_fitx_history[-self.avgFactor:], axis=0).astype(np.int), 0, 1279), 
                np.clip(np.average(self.right_fitx_history[-self.avgFactor:], axis=0).astype(np.int), 0, 1279))

        self.average_fit = (np.polyfit(self.ploty, self.average_fitx[0], 2),
                np.polyfit(self.ploty, self.average_fitx[1], 2))
        return self.get_average_fitx()

    def get_average_fit(self):
        return self.average_fit

    def get_average_fitx(self):
        return self.average_fitx
    
    def checkSanity(self, unchecked_lfit, unchecked_rfit):
        ploty = np.linspace(0, img_height-1, img_height)
        left_fitx = unchecked_lfit[0]*ploty**2 + unchecked_lfit[1]*ploty + unchecked_lfit[2]
        right_fitx = unchecked_rfit[0]*ploty**2 + unchecked_rfit[1]*ploty + unchecked_rfit[2]

        ## first time run
        if self.average_fitx == None:
            self.left_fitx_history = np.array([left_fitx])
            self.right_fitx_history = np.array([right_fitx])
            self.recalculate_average()
            return True

        unchecked_diff = right_fitx - left_fitx
        left_fitx_avg, right_fitx_avg = self.get_average_fitx()
        diff_avg = right_fitx_avg - left_fitx_avg
        check_status = np.absolute(diff_avg - unchecked_diff) > self.diffMargin
        false_index = check_status.nonzero()
        left_fitx[false_index] = left_fitx_avg[false_index]
        right_fitx[false_index] = right_fitx_avg[false_index]
        self.left_fitx_history = np.vstack((self.left_fitx_history, left_fitx))
        self.right_fitx_history = np.vstack((self.right_fitx_history, right_fitx))
        self.recalculate_average()

        # if 1/3 lane width is off compared with average make it fail
        return len(false_index) < img_height//3
        
    def log(self, msg, img, img2 = None):
        print(msg + ' Check {}/{}.jpg'.format(self.debugDir, self.debugCount))
        if self.enableDebug:
            self.saveDebugFig(img, img2)

    def saveDebugFig(self, img, img2):
        undist = cv2.undistort(img, mtx, dist)
        gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
        color_binary = color_threshold(undist)
        combined = color_binary
        combined_warped = perspective_transform(combined)
        original_warped = perspective_transform(undist)
        fig, axes = plt.subplots(3,3, figsize=(18,12))
        fig.tight_layout()
        images = [img, gray, combined, combined_warped, original_warped]
        titles = ['img', 'gray', 'combined', 'combined_warped', 'original_warped']
        if img2 != None:
            images.append(img2)
            titles.append('img2')
        for i, image in enumerate(images):
            if len(image.shape) == 3:
                fig.axes[i].imshow(image)
            else:
                fig.axes[i].imshow(image, cmap='gray')
            fig.axes[i].set_title(titles[i])

        plt.savefig('{}/{}.jpg'.format(tracker.debugDir, tracker.debugCount))
        plt.imsave('{}/original{}.jpg'.format(tracker.debugDir, tracker.debugCount), img)
        tracker.debugCount += 1

# manually selected points for warping. Make sure straightline look straight
src = np.float32([[(200, 720), (570, 470), (720, 470), (1130, 720)]])
dst = np.float32([[(320, 720), (320, 0), (920, 0), (920, 720)]])
# load calibration data
calibration_data = np.load('./calibration_data.npz')
mtx, dist = calibration_data['mtx'], calibration_data['dist']
# Test images
calibration_img = mpimg.imread('./camera_cal/calibration1.jpg')
img = mpimg.imread('./test_images/straight_lines1.jpg')
img_height = img.shape[0]
# retrieve_points_for_warping(img)

## process video
# video_in = 'project_video.mp4'
# video_in = 'challenge_video.mp4'
video_in = 'harder_challenge_video.mp4'
video_out = video_in.split('.')[0] + '_processed.' + video_in.split('.')[1]
clip = VideoFileClip(video_in)
tracker = Tracker()
tracker.enableDebug = True
tracker.debugDir = video_in.split('.')[0] + '_debug'
clip_processed = clip.fl_image(process_image)
clip_processed.write_videofile(video_out, audio=False)

## run thresholding on exemplar hard images
def test_threshold():
    for f in glob.glob('test_images/*'):
        print(f)
        img = mpimg.imread(f)
        color_binary = color_threshold(img)
        result_name = f.split('.')[0] + '_color.' + f.split('.')[1]
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # gradex_binary = abs_threshold(gray)
        # gradey_binary = abs_threshold(gray, 'y')
        # mag_binary = mag_threshold(gray)
        # dir_binary = dir_threshold(gray)
        # result_name = f.split('.')[0] + '_sobel.' + f.split('.')[1]
        # result = (gradex_binary & gradey_binary) | (mag_binary & dir_binary)
        mpimg.imsave(result_name, result, cmap='gray')

def test_pipeline():
    for f in glob.glob('test_images/*'):
        print(f)
        tracker.reset()
        img = mpimg.imread(f)
        result = process_image(img)
        result_name = f.split('.')[0] + '_processed.' + f.split('.')[1]
        mpimg.imsave(result_name, result, cmap='gray')

