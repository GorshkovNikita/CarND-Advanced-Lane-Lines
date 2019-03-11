import os
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from calibration import calibrate
from edges_detection import detect_edges
from warper import warp, rotation_matrix
from finding_lines import find_line_pixels
from line_fitting import fit_lines, plot_lines


def process_frame(frame_img, prev_left_line=None, prev_right_line=None):
    undistorted_img = cv2.undistort(frame_img, mtx, dst)
    edges_binary_img = detect_edges(undistorted_img)
    m = rotation_matrix()
    warped_image = warp(edges_binary_img, m)
    left_line_pixel_indexes, right_line_pixel_indexes = find_line_pixels(warped_image)
    left_fit_x, right_fit_x, ploty, left_fit, right_fit = \
        fit_lines(warped_image.shape, left_line_pixel_indexes, right_line_pixel_indexes)
    source_image_with_lines = plot_lines(frame_img, left_fit_x, right_fit_x, ploty)
    return source_image_with_lines


if __name__ == '__main__':
    """
      1. Camera calibration (once for the whole video). Returns camera matrix and distortion coefficients
      2. For each image:
        a. Undistort using data from the first step. Returns image
        b. Create gradient map for undistorted image using different thresholds. Returns binary image
        c. Warp this gradient map. Returns binary image
        d. Find lanes using window method (I prefer method with np.convolve). Returns lane pixels
        e. Fit pixels to polynomial of the second degree
        f. Calculate the radius of curvature and position of vehicle with respect to the center of the lane
        j. Plot lines on image
      3. Use described pipeline for video. Note that you need to use previously computed data for new frame (optional).
    """
    mtx, dst = calibrate()
    dir = './../test_images/'
    out_dir = './../out_images_with_lines/'
    for img_name in os.listdir(dir):
        print(img_name)
        img = mpimg.imread(dir + img_name)
        processed_frame = process_frame(img)
        mpimg.imsave(out_dir + img_name, processed_frame)
        plt.imshow(processed_frame)
        plt.show()

