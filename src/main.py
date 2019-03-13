import os
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import *

from calibration import calibrate
from thresholding import thresholded_binary_image
from warper import warp_and_sharpen, rotation_matrix
from finding_lines import find_line_pixels
from line_fitting import fit_lines, plot_lines, curve_radius, check_parallel
from line import Line


left_line = Line()
right_line = Line()


def process_frame(frame_img):
    undistorted_img = cv2.undistort(frame_img, mtx, dst)
    thresholded_binary_img = thresholded_binary_image(undistorted_img)
    m = rotation_matrix()
    warped_image = warp_and_sharpen(thresholded_binary_img, m)
    left_line_pixel_indexes, right_line_pixel_indexes = find_line_pixels(warped_image)
    left_fit_x, right_fit_x, ploty, left_fit, right_fit = \
        fit_lines(warped_image.shape, left_line_pixel_indexes, right_line_pixel_indexes)
    if not check_parallel(left_fit_x, right_fit_x):
        if left_line.last_pixels is not None:
            left_line_pixel_indexes = left_line.last_pixels
        if left_line.last_fit_c is not None:
            left_fit = left_line.last_fit_c
        if left_line.last_x is not None:
            left_fit_x = left_line.last_x

        if right_line.last_pixels is not None:
            right_line_pixel_indexes = right_line.last_pixels
        if right_line.last_fit_c is not None:
            right_fit = right_line.last_fit_c
        if right_line.last_x is not None:
            right_fit_x = right_line.last_x
    else:
        left_line.last_x = left_fit_x
        left_line.last_pixels = left_line_pixel_indexes
        left_line.last_fit_c = left_fit
        right_line.last_x = right_fit_x
        right_line.last_pixels = right_line_pixel_indexes
        right_line.last_fit_c = right_fit

    curve_rad = curve_radius(left_line_pixel_indexes, right_line_pixel_indexes)
    source_image_with_lines, offset = plot_lines(frame_img, left_fit_x, right_fit_x, ploty)
    cv2.putText(source_image_with_lines, 'Radius of curvature = ' + str(int(curve_rad)) + '(m)', (50, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, [255, 255, 255]
                )
    side = 'left' if offset < 0 else 'right'
    offset_text = 'Vehicle is ' + str(abs(round(offset, ndigits=2))) + '(m) ' + side + ' of center'
    cv2.putText(source_image_with_lines, offset_text, (50, 100),
                cv2.FONT_HERSHEY_COMPLEX, 1, [255, 255, 255]
                )

    return source_image_with_lines


def process_video(process_image):
    filename = 'project_video'
    white_output = './../output_videos/' + filename + '.mp4'
    clip = VideoFileClip('./../' + filename + '.mp4')
    white_clip = clip.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)


def cut_video(start_second, end_second):
    filename = 'project_video'
    clip = VideoFileClip('./../' + filename + '.mp4')
    cut_clip = clip.subclip(start_second, end_second)
    cut_clip.write_videofile('./../hard_cut_' + filename + '.mp4')


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
    # cut_video(38, 43)
    process_video(process_frame)
    # dir = './../test_images/'
    # out_dir = './../out_binary_images/'
    # for img_name in os.listdir(dir):
    #     print(img_name)
    #     img_name = 'test9.jpg'
    #     img = mpimg.imread(dir + img_name)
    #     processed_frame = process_frame(img)
    #     # mpimg.imsave(out_dir + img_name, processed_frame, cmap='gray')
    #     plt.imshow(processed_frame)
    #     plt.show()
