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
    return mtx, dist

## Run once and save the cameraMatrix and distCoeffs
# mtx, dist = get_mtx_dist()
# np.savez('./calibration_data.npz', mtx=mtx, dist=dist)

calibration_data = np.load('./calibration_data.npz')
mtx, dist = calibration_data['mtx'], calibration_data['dist']

## Test undistort
# test_img = mpimg.imread('./camera_cal/calibration1.jpg')
# undist = cv2.undistort(test_img, mtx, dist)
# fig, axes = plt.subplots(1,2,figsize=(10,5))
# axes[0].imshow(test_img)
# axes[1].imshow(undist)
# plt.show()


img = mpimg.imread('./test_images/straight_lines1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
kernel_size = 5
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, kernel_size)
absolute_sobelx = np.absolute(sobelx)
scaled_sobelx = 255*absolute_sobelx/np.max(absolute_sobelx)
sobelx_binary = np.zeros_like(gray)
sobelx_thresh = (30, 100)
sobelx_binary[(scaled_sobelx > sobelx_thresh[0]) & (scaled_sobelx < sobelx_thresh[1])] = 1

hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
schannel = hls[:,:,2]
schannel_binary = np.zeros_like(schannel)
thresh = (30,100)
schannel_binary[(schannel > thresh[0]) & (schannel < thresh[1])] = 1

fig, axes = plt.subplots(1,2,figsize=(10,6))
axes[0].imshow(sobelx_binary, cmap='gray')
axes[1].imshow(schannel_binary, cmap='gray')
plt.show()
