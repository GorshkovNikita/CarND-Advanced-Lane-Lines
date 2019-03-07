import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def sobel_binary(image, thresh=(50, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    binary = np.zeros_like(scaled_sobel, dtype=float)
    binary[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary


def hls_binary(image, thresh=(150, 255), axis=2):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    channel = hls[:, :, axis]
    binary = np.zeros_like(channel, dtype=float)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary


if __name__ == '__main__':
    img = mpimg.imread('./../test_images/test2.jpg')
    s_binary = hls_binary(img)
    sobel_binary = sobel_binary(img)
    stacked_img = np.dstack((np.zeros_like(s_binary), s_binary, sobel_binary)) * 255
    plt.imshow(stacked_img)
    plt.show()

