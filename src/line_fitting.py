import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from warper import rotation_matrix

my = 30 / 720  # meters per pixel in y dimension
mx = 3.7 / 700  # meters per pixel in x dimension


def fit_lines(img_shape, left_line_pixel_indices, right_line_pixel_indices):
    l_fit_coef = np.polyfit(left_line_pixel_indices[0], left_line_pixel_indices[1], 2)
    r_fit_coef = np.polyfit(right_line_pixel_indices[0], right_line_pixel_indices[1], 2)
    y = np.linspace(0, img_shape[0] - 1, img_shape[0])

    l_fit_x = l_fit_coef[0] * y ** 2 + l_fit_coef[1] * y + l_fit_coef[2]
    r_fit_x = r_fit_coef[0] * y ** 2 + r_fit_coef[1] * y + r_fit_coef[2]

    return l_fit_x, r_fit_x, y, l_fit_coef, r_fit_coef


def curve_radius(left_line_pixel_indices, right_line_pixel_indices):
    y_val = np.max(left_line_pixel_indices[0])
    l_fit = np.polyfit(left_line_pixel_indices[0] * my, left_line_pixel_indices[1] * mx, 2)
    r_fit = np.polyfit(right_line_pixel_indices[0] * my, right_line_pixel_indices[1] * mx, 2)
    left_curve_radius = ((1 + (2 * l_fit[0] * y_val * my + l_fit[1]) ** 2) ** (3 / 2)) / np.absolute(2 * l_fit[0])
    right_curve_radius = ((1 + (2 * r_fit[0] * y_val * my + r_fit[1]) ** 2) ** (3 / 2)) / np.absolute(2 * r_fit[0])
    return (left_curve_radius + right_curve_radius) / 2


def plot_lines(src_img, l_fit_x, r_fit_x, y):
    m_inv = rotation_matrix(True)
    right_x = np.array(r_fit_x).astype(int)
    right_indices = np.where((right_x < src_img.shape[1]) & (right_x >= 0))[0]
    right_filtered_x = right_x[right_indices]

    left_x = np.array(l_fit_x).astype(int)
    left_indices = np.where((left_x < src_img.shape[1]) & (left_x >= 0))[0]
    left_filtered_x = left_x[left_indices]

    left_pts = np.dstack((left_filtered_x, np.array(y).astype(int)[left_indices]))[0]
    right_pts = np.dstack((right_filtered_x, np.array(y).astype(int)[right_indices]))[0]

    image_with_curves = np.zeros_like(src_img)
    cv2.polylines(image_with_curves, np.array([left_pts]), False, [255, 0, 0], thickness=25)
    cv2.polylines(image_with_curves, np.array([right_pts]), False, [0, 0, 255], thickness=25)
    points1 = np.array([[[xi, yi]] for xi, yi in left_pts]).astype(np.int32)
    points2 = np.array([[[xi, yi]] for xi, yi in right_pts]).astype(np.int32)
    points2 = np.flipud(points2)
    points = np.concatenate((points1, points2))
    cv2.fillPoly(image_with_curves, [points], [0, 120, 0])
    warped = cv2.warpPerspective(image_with_curves, m_inv, (src_img.shape[1], src_img.shape[0]))
    left_line_pos = np.argmax(warped[src_img.shape[0] - 10, :, 0])
    right_line_pos = np.argmax(warped[src_img.shape[0] - 10, :, 2])
    real_pos = src_img.shape[1] / 2
    ideal_pos = left_line_pos + (right_line_pos - left_line_pos) / 2
    offset = ((real_pos - ideal_pos) * mx)
    return cv2.addWeighted(src_img, 1.0, warped, 0.7, 1.0), offset


def check_parallel(left_fit_x, right_fit_x):
    diff = right_fit_x - left_fit_x
    return abs(np.max(diff) - np.min(diff)) < 250


if __name__ == '__main__':
    import pickle
    filename = 'test7'
    image = mpimg.imread('./../out_binary_images/' + filename + '.jpg')
    f = open('./../output_lines_pixels/' + filename, 'rb')
    left_pixel_indices, right_pixel_indices = pickle.load(f)
    left_fit_x, right_fit_x, ploty, left_fit_coef, right_fit_coef = \
        fit_lines(image.shape, left_pixel_indices, right_pixel_indices)
    print(check_parallel(left_fit_x, right_fit_x))
    curve_rad = curve_radius(left_pixel_indices, right_pixel_indices)
    # print(curve_rad)

    source_image = mpimg.imread('./../test_images/' + filename + '.jpg')
    source_image_with_lines, offset = plot_lines(source_image, left_fit_x, right_fit_x, ploty)
    cv2.putText(source_image_with_lines, 'Radius of curvature = ' + str(int(curve_rad)) + '(m)', (50, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, [255, 255, 255]
                )
    side = 'left' if offset < 0 else 'right'
    offset_text = 'Vehicle is ' + str(abs(round(offset, ndigits=2))) + '(m) ' + side + ' of center'
    cv2.putText(source_image_with_lines, offset_text, (50, 100),
                cv2.FONT_HERSHEY_COMPLEX, 1, [255, 255, 255]
                )
    # plt.imshow(source_image_with_lines)
    # plt.imsave('./../examples/example_output.jpg', source_image_with_lines)
    # plt.plot(left_fit_x, ploty, color='red')
    # plt.plot(right_fit_x, ploty, color='red')
    # plt.imshow(image, cmap='gray')
    # plt.show()
