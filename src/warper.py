import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# https://www.image-map.net/
def rotation_matrix(inv=False):
    # it is possible to cache it for efficiency
    src = np.float32([[210, 718], [601, 448], [680, 446], [1106, 718]])
    dst = np.float32([[250, 720], [250, 0], [1000, 0], [1000, 720]])
    if inv:
        return cv2.getPerspectiveTransform(dst, src)
    else:
        return cv2.getPerspectiveTransform(src, dst)


def warp(image, m):
    return cv2.warpPerspective(image, m, (image.shape[1], image.shape[0]))


def warp_and_sharpen(image, m):
    # sharpening image after warping
    return np.rint(warp(image, m)).astype(int)


if __name__ == '__main__':
    from thresholding import thresholded_binary_image
    src = np.float32([[210, 718], [601, 448], [680, 446], [1106, 718]])
    dst = np.float32([[250, 720], [250, 0], [1000, 0], [1000, 720]])
    img = mpimg.imread('./../test_images/straight_lines1.jpg')
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # thresholded_image = thresholded_binary_image(img)
    m = rotation_matrix()
    # warped_img = warp_and_convert_to_binary(thresholded_binary_image(img), m)
    warped_img = warp(img, m)
    cv2.polylines(img, np.int32([src]), True, [255, 0, 0], thickness=3)
    plt.imshow(img, cmap='gray')
    plt.show()
    cv2.polylines(warped_img, np.int32([dst]), True, [255, 0, 0], thickness=3)
    plt.imshow(warped_img, cmap='gray')
    plt.show()
