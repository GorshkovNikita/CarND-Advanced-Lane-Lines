import numpy as np
import matplotlib.pyplot as plt
import cv2


def window_mask(img, layer, window_width, window_height, center):
    image_height = img.shape[0]
    mask = np.zeros_like(img)
    mask[
        int(image_height - (layer + 1) * window_height): int(image_height - layer * window_height),
        int(center - window_width / 2): int(center + window_width / 2)
    ] = 1
    return mask


def find_line_pixels(img):
    window_centroids = []
    image_width = img.shape[1]
    image_height = img.shape[0]
    nwindows = 9
    window_height = image_height / nwindows
    window_width = 50
    window = np.ones(window_width)
    margin = 100
    left_line_x_pixel_indexes = np.array([], dtype=np.int64)
    left_line_y_pixel_indexes = np.array([], dtype=np.int64)
    right_line_x_pixel_indexes = np.array([], dtype=np.int64)
    right_line_y_pixel_indexes = np.array([], dtype=np.int64)

    left_bottom_image_part = img[int(3 * image_height / 4):, :int(image_width / 2)]
    right_bottom_image_part = img[int(3 * image_height / 4):, int(image_width / 2):]
    # number of activated pixels in right and left parts of image bottom
    left_pixels = np.sum(left_bottom_image_part, axis=0)
    right_pixels = np.sum(right_bottom_image_part, axis=0)
    # find coordinate with maximum number of neighbouring pixels
    l_center = np.argmax(np.convolve(window, left_pixels, mode='same'))
    r_center = np.argmax(np.convolve(window, right_pixels, mode='same')) + int(image_width / 2)

    for layer in range(0, nwindows):
        image_layer = image[int(image_height - (layer + 1) * window_height) : int(image_height - layer * window_height), :]
        # number of activated pixels for each coordinate in image layer
        layer_pixels = np.sum(image_layer, axis=0)
        # find convolution of the same length as image layer
        conv_signal = np.convolve(window, layer_pixels, mode='same')

        # find indices for search area
        left_min_index = max(l_center - margin, 0)
        left_max_index = min(l_center + margin, image_width)

        right_min_index = max(r_center - margin, 0)
        right_max_index = min(r_center + margin, image_width)

        # find index with maximum neighbouring pixels in search area
        l_center = np.argmax(conv_signal[left_min_index:left_max_index]) + left_min_index
        r_center = np.argmax(conv_signal[right_min_index:right_max_index]) + right_min_index

        # todo: если вообще не найдено ни одного пикселя, или меньше какого-то порога,
        # todo: нужно это проверять и не добавлять наверно в центроиды или добавить что-то типа (-1, -1)
        window_centroids.append((l_center, r_center))

        # find activated indices inside window
        l_pixels = np.nonzero(cv2.bitwise_and(img, window_mask(img, layer, window_width, window_height, l_center)))
        r_pixels = np.nonzero(cv2.bitwise_and(img, window_mask(img, layer, window_width, window_height, r_center)))

        # add these indices to resulting array of all indices of corresponding line
        left_line_x_pixel_indexes = np.concatenate((left_line_x_pixel_indexes, l_pixels[0]))
        left_line_y_pixel_indexes = np.concatenate((left_line_y_pixel_indexes, l_pixels[1]))
        right_line_x_pixel_indexes = np.concatenate((right_line_x_pixel_indexes, r_pixels[0]))
        right_line_y_pixel_indexes = np.concatenate((right_line_y_pixel_indexes, r_pixels[1]))

    # return tuple of tuples with indices (x, y) of line pixels
    return (left_line_x_pixel_indexes, left_line_y_pixel_indexes),\
           (right_line_x_pixel_indexes, right_line_y_pixel_indexes)


if __name__ == '__main__':
    filename = 'straight_lines2'
    image = np.rint(cv2.imread('./../out_binary_images/' + filename + '.jpg', 0) / 255).astype(int)
    left_line_pixel_indexes, right_line_pixel_indexes = find_line_pixels(image)
    import pickle
    f = open('./../output_lines_pixels/' + filename, 'wb')
    pickle.dump((left_line_pixel_indexes, right_line_pixel_indexes), f, protocol=0)
    scaled = np.uint8(255 * image / np.max(image))
    output_image = np.stack((scaled, scaled, scaled), axis=2)
    output_image[left_line_pixel_indexes[0], left_line_pixel_indexes[1], :] = [255, 0, 0]
    output_image[right_line_pixel_indexes[0], right_line_pixel_indexes[1], :] = [0, 255, 0]

    plt.imshow(output_image)
    plt.show()

