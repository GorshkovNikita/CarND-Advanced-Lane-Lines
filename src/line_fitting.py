import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2


def fit_lines(img_shape, left_line_pixel_indices, right_line_pixel_indices):
    my = 30 / 720  # meters per pixel in y dimension
    mx = 3.7 / 700  # meters per pixel in x dimension
    left_fit = np.polyfit(left_line_pixel_indices[0] * my, left_line_pixel_indices[1] * mx, 2)
    right_fit = np.polyfit(right_line_pixel_indices[0] * my, right_line_pixel_indices[1] * mx, 2)
    y = np.linspace(0, img_shape[0] - 1, img_shape[0]) * my

    l_fit_x = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    r_fit_x = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]

    # in order to display these lines on source image i need to unwarp it
    return l_fit_x / mx, r_fit_x / mx, y, left_fit, right_fit


def curve_radius(l_fit, r_fit, y):
    y_val = np.max(y)
    left_curve_radius = ((1 + (2 * l_fit[0] * y_val + l_fit[1]) ** 2) ** (3 / 2)) / np.absolute(2 * l_fit[0])
    right_curve_radius = ((1 + (2 * r_fit[0] * y_val + r_fit[1]) ** 2) ** (3 / 2)) / np.absolute(2 * r_fit[0])
    return left_curve_radius, right_curve_radius


if __name__ == '__main__':
    import pickle
    from warper import rotation_matrix

    filename = 'test6'
    image = mpimg.imread('./../out_binary_images/' + filename + '.jpg')
    m_inv = rotation_matrix(True)
    transformed = cv2.warpPerspective(image, m_inv, (image.shape[1], image.shape[0]))
    f = open('./../output_lines_pixels/' + filename, 'rb')
    left_pixel_indices, right_pixel_indices = pickle.load(f)
    left_fit_x, right_fit_x, ploty, left_fit, right_fit = \
        fit_lines(image.shape, left_pixel_indices, right_pixel_indices)
    print(curve_radius(left_fit, right_fit, ploty))

    image_with_curves = np.zeros_like(image)
    image_with_curves[np.array(ploty / (30 / 720)).astype(int), np.array(left_fit_x).astype(int)] = [255, 0, 0, 255]
    # todo: здесь индексы выходят за рамки (720, 1280), поэтому надо как-то обрезать наверно, но не факт
    #  надо глянуть, чем он отличается от левой полосы
    # image_with_curves[np.array(ploty / (30 / 720)).astype(int), np.array(right_fit_x).astype(int)] = [255, 0, 0, 255]

    # plt.plot(left_fit_x, ploty / (30 / 720), color='red')
    # plt.plot(right_fit_x, ploty / (30 / 720), color='red')

    plt.imshow(mpimg.imread('./../test_images/test6.jpg'))
    plt.imshow(cv2.warpPerspective(image_with_curves, m_inv, (image.shape[1], image.shape[0])), cmap='gray')
    plt.show()
