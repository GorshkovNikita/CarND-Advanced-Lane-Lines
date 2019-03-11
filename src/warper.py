import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rotation_matrix(inv=False):
    # it is possible to cache it for efficiency
    src = np.float32([[210, 718], [601, 448], [680, 446], [1106, 718]])
    dst = np.float32([[250, 720], [250, 0], [1000, 0], [1000, 720]])
    if inv:
        return cv2.getPerspectiveTransform(dst, src)
    else:
        return cv2.getPerspectiveTransform(src, dst)


def warp(image, m):
    # https://www.image-map.net/
    return np.rint(cv2.warpPerspective(image, m, (image.shape[1], image.shape[0]))).astype(int), m_inv


if __name__ == '__main__':
    from edges_detection import detect_edges
    img = mpimg.imread('./../test_images/test5.jpg')
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = detect_edges(img)
    m = rotation_matrix()
    warped_img = warp(edges, m)
    plt.imshow(warped_img, cmap='gray')
    plt.show()
