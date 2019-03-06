import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def detect_edges(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    return S


if __name__ == '__main__':
    img = detect_edges(mpimg.imread('./../test_images/test1.jpg'))
    plt.imshow(img, cmap='gray')
    plt.show()

