import numpy as np
import cv2 as cv
import math
import sys, os

calculate_homography = __import__("1").calculate_homography
WINDOW_NAME = "window"


def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param[0].append([x, y])


def sort_4_points(points):
    center_x, center_y = np.mean(points, axis=0)
    sorted_points = np.empty_like(points)
    for point in points:
        if point[0] < center_x and point[1] < center_y:
            sorted_points[0] = point
        elif point[0] > center_x and point[1] < center_y:
            sorted_points[1] = point
        elif point[0] < center_x and point[1] > center_y:
            sorted_points[2] = point
        elif point[0] > center_x and point[1] > center_y:
            sorted_points[3] = point
    return sorted_points


def get_points(img):
    points_add = []
    cv.namedWindow(WINDOW_NAME)
    cv.setMouseCallback(WINDOW_NAME, on_mouse, [points_add])

    while True:
        img_ = img.copy()
        for i, p in enumerate(points_add):
            # draw points on img_
            cv.circle(img_, tuple(p), 2, (0, 255, 0), -1)
        cv.imshow(WINDOW_NAME, img_)

        key = cv.waitKey(20) % 0xFF
        if key == 27 or len(points_add) == 4:
            break  # exist when pressing ESC
    cv.destroyAllWindows()

    return sort_4_points(points_add)


def backward_warping(img, invH, w, h):
    warp_img = np.zeros((h, w, 3), np.uint8)
    for i in range(h):
        for j in range(w):
            x, y, z = invH.dot(np.array([i, j, 1]))
            x /= z
            y /= z
            warp_img[i, j] = bilinear_interpolation(img, x, y)

    return warp_img


def bilinear_interpolation(img, x, y):
    x_floor, x_ceil = math.floor(x), math.ceil(x)
    y_floor, y_ceil = math.floor(y), math.ceil(y)

    a, b, c, d = (
        img[y_floor, x_floor],
        img[y_floor, x_ceil],
        img[y_ceil, x_ceil],
        img[y_ceil, x_floor],
    )
    wa = (x_ceil - x) * (y_ceil - y)
    wb = (x - x_floor) * (y_ceil - y)
    wc = (x - x_floor) * (y - y_floor)
    wd = (x_ceil - x) * (y - y_floor)
    return wa * a + wb * b + wc * c + wd * d


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[USAGE] python3 2.py [IMAGE PATH]")
        sys.exit(1)
    img_path = sys.argv[1]
    img = cv.imread(img_path)
    print("Image shape:", img.shape)
    height, width = img.shape[0], img.shape[1]
    points_of_projection = np.array(
        [[0, 0], [0, width - 1], [height - 1, 0], [height - 1, width - 1]]
    )
    points = get_points(img)  # [[642, 898], [2357, 876], [285, 3549], [2736, 3523]]
    print("Selected corners:\n", points)

    H = calculate_homography(points, points_of_projection)
    print("Calulated Homography:\n", H)
    warp_img = backward_warping(img, np.linalg.inv(H), width, height)
    cv.imwrite(
        img_path.rsplit(".", 1)[0] + "(warped)." + img_path.rsplit(".", 1)[1], warp_img
    )
    cv.imshow("warped", warp_img)
    cv.waitKey(0)
