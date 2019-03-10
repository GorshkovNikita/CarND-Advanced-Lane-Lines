import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def warp(image):
    # https://www.image-map.net/
    src = np.float32([[210, 718], [601,448], [680,446], [1106,718]])
    dst = np.float32([[250, 720], [250, 0], [1000, 0], [1000, 720]])
    # image = cv2.line(image, tuple(src[0]), tuple(src[1]), [255, 0, 0], thickness=2)
    # image = cv2.line(image, tuple(src[1]), tuple(src[2]), [255, 0, 0], thickness=2)
    # image = cv2.line(image, tuple(src[2]), tuple(src[3]), [255, 0, 0], thickness=2)
    # image = cv2.line(image, tuple(src[3]), tuple(src[0]), [255, 0, 0], thickness=2)
    # plt.imshow(image)
    # plt.show()
    # M = cv2.getPerspectiveTransform(src, dst)
    m = cv2.getPerspectiveTransform(src, dst)
    return np.rint(cv2.warpPerspective(image, m, (1280, 720))).astype(int)


if __name__ == '__main__':
    from edges_detection import detect_edges
    img = mpimg.imread('./../test_images/test5.jpg')
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = detect_edges(img)
    print(np.unique(edges))
    warped_img = warp(edges)
    print(warped_img.shape)
    print(type(warped_img))
    print(np.unique(np.rint(warped_img)))
    plt.imshow(np.rint(warped_img), cmap='gray')
    plt.show()
