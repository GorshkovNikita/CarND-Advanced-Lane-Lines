import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def sobel_binary(image, thresh=(20, 300)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    binary = np.zeros_like(scaled_sobel, dtype=float)
    binary[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary


def hls_binary(image, thresh=(120, 255), axis=2):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    channel = hls[:, :, axis]
    binary = np.zeros_like(channel, dtype=float)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary


def thresholded_binary_image(image):
    s_binary = hls_binary(image)
    sobel_b = sobel_binary(image)
    result_binary = np.zeros_like(s_binary)
    result_binary[(sobel_b == 1) | (s_binary == 1)] = 1
    return result_binary


if __name__ == '__main__':
    img = mpimg.imread('./../test_images/test4.jpg')
    # s_binary = hls_binary(img)
    # sobel_binary = sobel_binary(img)
    # stacked_img = np.uint8(np.dstack((np.zeros_like(s_binary), s_binary, sobel_binary)) * 255)
    # plt.imshow(sobel_binary(img), cmap='gray')
    # plt.imshow(hls_binary(img, axis=1), cmap='gray')
    plt.imshow(thresholded_binary_image(img), cmap='gray')
    # plt.imshow(img)
    plt.show()
