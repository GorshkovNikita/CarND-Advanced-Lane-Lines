import calibration

if __name__ == '__main__':
    """
      1. Camera calibration (once for the whole video). Returns camera matrix and distortion coefficients
      2. For each image:
        a. Undistort using data from the first step. Returns image
        b. Create gradient map for undistorted image using different thresholds. Returns binary image
        c. Warp this gradient map. Returns binary image
        d. Find lanes using window method (I prefer method with np.convolve). Returns lane pixels
        e. Fit pixels to polynomial of the second degree
        f. Calculate the radius of curvature and position of vehicle with respect to center
        j. Plot lines on image
    """
    calibration.some_func()
