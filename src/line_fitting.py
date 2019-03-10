import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def fit_lines(img_shape, left_line_pixel_indices, right_line_pixel_indices):
    left_fit = np.polyfit(left_line_pixel_indices[0], left_line_pixel_indices[1], 2)
    right_fit = np.polyfit(right_line_pixel_indices[0], right_line_pixel_indices[1], 2)
    y = np.linspace(0, img_shape[0] - 1, img_shape[0])

    l_fit_x = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    r_fit_x = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]

    return l_fit_x, r_fit_x, y


if __name__ == '__main__':
    import pickle
    filename = 'test4'
    image = mpimg.imread('./../out_binary_images/' + filename + '.jpg')
    f = open('./../output_lines_pixels/' + filename, 'rb')
    left_pixel_indices, right_pixel_indices = pickle.load(f)
    left_fit_x, right_fit_x, ploty = fit_lines(image.shape, left_pixel_indices, right_pixel_indices)

    plt.plot(left_fit_x, ploty, color='red')
    plt.plot(right_fit_x, ploty, color='red')

    plt.imshow(image, cmap='gray')
    plt.show()

