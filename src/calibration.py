import cv2
import os
import numpy as np
import matplotlib.image as mpimg


def calibrate():
    img_points = []
    object_points = []
    objp = np.zeros((9*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    root_dir = './../camera_cal/'
    image_paths = os.listdir(root_dir)
    for path in image_paths:
        image = mpimg.imread(root_dir + path)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            img_points.append(corners)
            object_points.append(objp)
            shape = gray.shape[::-1]
    retval, mtx, dst, rvecs, tvecs = cv2.calibrateCamera(object_points, img_points, shape, None, None)
    return mtx, dst


if __name__ == '__main__':
    # undistorted image test
    import matplotlib.pyplot as plt
    mtx, dst = calibrate()
    image = cv2.imread('../camera_cal/calibration2.jpg')
    plt.imshow(image)
    plt.show()
    undistorted_img = cv2.undistort(image, mtx, dst)
    plt.imshow(undistorted_img)
    plt.show()
