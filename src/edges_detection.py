import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def sobel_binary(image, thresh=(20, 100)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    binary = np.zeros_like(scaled_sobel, dtype=float)
    binary[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary


def hls_binary(image, thresh=(170, 255), axis=2):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    channel = hls[:, :, axis]
    binary = np.zeros_like(channel, dtype=float)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary


def detect_edges(image):
    s_binary = hls_binary(image)
    sobel_b = sobel_binary(image)
    result_binary = np.zeros_like(s_binary)
    result_binary[(sobel_b == 1) | (s_binary == 1)] = 1
    return result_binary


if __name__ == '__main__':
    img = mpimg.imread('./../test_images/test5.jpg')
    # s_binary = hls_binary(img)
    # sobel_binary = sobel_binary(img)
    # stacked_img = np.uint8(np.dstack((np.zeros_like(s_binary), s_binary, sobel_binary)) * 255)
    # plt.imshow(stacked_img)
    plt.imshow(detect_edges(img), cmap='gray')
    plt.show()
