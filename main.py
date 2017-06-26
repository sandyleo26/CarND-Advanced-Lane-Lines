import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob

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

def sobel_threshold(img, thresh=(30, 100), direction='x', kernel_size=5):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel_size = 5
    sobel = None
    if direction=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, kernel_size)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, kernel_size)

    absolute_sobel = np.absolute(sobel)
    scaled_sobel = 255*absolute_sobel/np.max(absolute_sobel)
    sobel_binary = np.zeros_like(gray)
    sobel_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return sobol_binary

def color_threshold(img, thresh=(30, 100)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    schannel = hls[:,:,2]
    schannel_binary = np.zeros_like(schannel)
    thresh = (30,100)
    schannel_binary[(schannel > thresh[0]) & (schannel < thresh[1])] = 1

    ## Test sobel threshold and color threshold
    # fig, axes = plt.subplots(1,2,figsize=(10,6))
    # axes[0].imshow(schannel, cmap='gray')
    # axes[1].imshow(schannel_binary, cmap='gray')
    # plt.show()
    return schannel_binary

## find 4 points for perspective transform
def retrieve_points_for_warping():

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
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    src = np.array([(302,650), (1002,652), (765,502), (525,503)], np.float32)
    dst = np.array([(302,650), (1002,650), (1002,502), (302,502)], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(gray, M, gray.shape[::-1], cv2.INTER_LINEAR)

    ## Test perspective transform
    # fig, axes = plt.subplots(1,2,figsize=(10,6))
    # axes[0].imshow(gray)
    # axes[1].imshow(warped)
    # plt.show()
    return warped

## Test images
calibration_img = mpimg.imread('./camera_cal/calibration1.jpg')
straight_lines1_img = mpimg.imread('./test_images/straight_lines1.jpg')

calibration_data = np.load('./calibration_data.npz')
mtx, dist = calibration_data['mtx'], calibration_data['dist']

## Test undistort
# undist = cv2.undistort(img, mtx, dist)
# fig, axes = plt.subplots(1,2,figsize=(10,5))
# axes[0].imshow(calibration_img)
# axes[1].imshow(undist)
# plt.show()



