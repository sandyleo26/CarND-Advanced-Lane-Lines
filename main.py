import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob

def calibrate(img):

    return img

def get_mtx_dist():
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
    # undist = cv2.undistort(img, mtx, dist)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    # ax1.imshow(img)
    # ax2.imshow(undist)
    # plt.show()
    # break

    return mtx, dist

mtx, dist = get_mtx_dist()
