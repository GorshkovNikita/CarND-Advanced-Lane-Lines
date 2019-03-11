import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from warper import rotation_matrix

# todo: поднастроить
my = 30 / 720  # meters per pixel in y dimension
mx = 3.7 / 700  # meters per pixel in x dimension


def fit_lines(img_shape, left_line_pixel_indices, right_line_pixel_indices):
    l_fit = np.polyfit(left_line_pixel_indices[0] * my, left_line_pixel_indices[1] * mx, 2)
    r_fit = np.polyfit(right_line_pixel_indices[0] * my, right_line_pixel_indices[1] * mx, 2)
    y = np.linspace(0, img_shape[0] - 1, img_shape[0]) * my

    l_fit_x = l_fit[0] * y ** 2 + l_fit[1] * y + l_fit[2]
    r_fit_x = r_fit[0] * y ** 2 + r_fit[1] * y + r_fit[2]

    # in order to display these lines on source image i need to unwarp it
    return l_fit_x / mx, r_fit_x / mx, y, l_fit, r_fit


def curve_radius(l_fit, r_fit, y):
    y_val = np.max(y)
    left_curve_radius = ((1 + (2 * l_fit[0] * y_val + l_fit[1]) ** 2) ** (3 / 2)) / np.absolute(2 * l_fit[0])
    right_curve_radius = ((1 + (2 * r_fit[0] * y_val + r_fit[1]) ** 2) ** (3 / 2)) / np.absolute(2 * r_fit[0])
    return left_curve_radius, right_curve_radius


def plot_lines(src_img, l_fit_x, r_fit_x, y):
    m_inv = rotation_matrix(True)
    right_x = np.array(r_fit_x).astype(int)
    right_indices = np.where(right_x < src_img.shape[1])[0]
    right_filtered_x = right_x[right_indices]

    left_x = np.array(l_fit_x).astype(int)
    left_indices = np.where(left_x < src_img.shape[1])[0]
    left_filtered_x = left_x[left_indices]

    left_pts = np.dstack((left_filtered_x, np.array(y / my).astype(int)[left_indices]))[0]
    right_pts = np.dstack((right_filtered_x, np.array(y / my).astype(int)[right_indices]))[0]

    image_with_curves = np.zeros_like(src_img)
    cv2.polylines(image_with_curves, np.array([left_pts, right_pts]), False, [255, 0, 0], thickness=25)
    warped = cv2.warpPerspective(image_with_curves, m_inv, (src_img.shape[1], src_img.shape[0]))
    return cv2.addWeighted(src_img, 1.0, warped, 0.7, 1.0)


if __name__ == '__main__':
    import pickle

    filename = 'test7'
    image = mpimg.imread('./../out_binary_images/' + filename + '.jpg')
    f = open('./../output_lines_pixels/' + filename, 'rb')
    left_pixel_indices, right_pixel_indices = pickle.load(f)
    left_fit_x, right_fit_x, ploty, left_fit, right_fit = \
        fit_lines(image.shape, left_pixel_indices, right_pixel_indices)
    print(curve_radius(left_fit, right_fit, ploty))

    source_image = mpimg.imread('./../test_images/' + filename + '.jpg')
    plt.imshow(plot_lines(source_image, left_fit_x, right_fit_x, ploty))
    plt.show()
