import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def fit_lines(img_shape, left_line_pixel_indices, right_line_pixel_indices):
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    left_fit = np.polyfit(left_line_pixel_indices[0] * ym_per_pix, left_line_pixel_indices[1] * xm_per_pix, 2)
    right_fit = np.polyfit(right_line_pixel_indices[0] * ym_per_pix, right_line_pixel_indices[1] * xm_per_pix, 2)
    y = np.linspace(0, img_shape[0] - 1, img_shape[0]) * ym_per_pix

    l_fit_x = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    r_fit_x = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]

    # in order to display these lines on source image i need to unwarp it
    return l_fit_x / xm_per_pix, r_fit_x / xm_per_pix, y, left_fit, right_fit


def curve_radius(l_fit, r_fit, y):
    y_val = np.max(y)
    left_curve_radius = ((1 + (2 * l_fit[0] * y_val + l_fit[1]) ** 2) ** (3 / 2)) / np.absolute(2 * l_fit[0])
    right_curve_radius = ((1 + (2 * r_fit[0] * y_val + r_fit[1]) ** 2) ** (3 / 2)) / np.absolute(2 * r_fit[0])
    return left_curve_radius, right_curve_radius


if __name__ == '__main__':
    import pickle
    filename = 'test7'
    image = mpimg.imread('./../out_binary_images/' + filename + '.jpg')
    f = open('./../output_lines_pixels/' + filename, 'rb')
    left_pixel_indices, right_pixel_indices = pickle.load(f)
    left_fit_x, right_fit_x, ploty, left_fit, right_fit = fit_lines(image.shape, left_pixel_indices, right_pixel_indices)
    print(curve_radius(left_fit, right_fit, ploty))

    plt.plot(left_fit_x, ploty / (30 / 720), color='red')
    plt.plot(right_fit_x, ploty / (30 / 720), color='red')

    plt.imshow(image, cmap='gray')
    plt.show()

