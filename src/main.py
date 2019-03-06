from edges_detection import detect_edges
from calibration import calibrate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


def process_frame(frame_img, prev_left_line=None, prev_right_line=None):
    undistorted_img = cv2.undistort(frame_img, mtx, dst)
    # edges_binary_img = detect_edges(undistorted_img)
    return undistorted_img


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
    img = mpimg.imread('./../test_images/test1.jpg')
    processed_frame = process_frame(img)
    plt.imshow(processed_frame)
    plt.show()

