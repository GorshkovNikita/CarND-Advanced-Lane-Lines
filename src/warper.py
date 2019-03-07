import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def warp(image):
    # 190,720,590,450,700,450,1130,720
    src = np.float32([[190, 720], [590, 450], [700, 450], [1130, 720]])
    dst = np.float32([[100, 720], [100, 0], [620, 0], [620, 720]])
    # M = cv2.getPerspectiveTransform(src, dst)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, m, (720, 720))


if __name__ == '__main__':
    img = mpimg.imread('./../test_images/test5.jpg')
    plt.imshow(warp(img))
    plt.show()
