import numpy as np
import cv2 as cv


def get_sift_correspondences(img1, img2, k):
    """
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    """
    # sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()  # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:k]

    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    # """ draw correspondences """
    # img_draw_match = cv.drawMatches(
    #     img1,
    #     kp1,
    #     img2,
    #     kp2,
    #     good_matches,
    #     None,
    #     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    # )
    # cv.imshow(f"match", img_draw_match)
    # cv.waitKey(0)
    return points1, points2


def get_average_dist_to_origin(points):
    dist = (points - [0, 0]) ** 2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)

    return np.mean(dist)


def normalize_image_points(image):
    """
    Input: 2D list with x,y image points
    Output:
    """

    # print()
    # print("Normalizing data using similarity matrix...")

    image = np.array(image)
    mean, std = np.mean(image, 0), np.std(image)

    # define similarity transformation
    # no rotation, scaling using sdv and setting centroid as origin
    Transformation = np.array(
        [[std / np.sqrt(2), 0, mean[0]], [0, std / np.sqrt(2), mean[1]], [0, 0, 1]]
    )

    # apply transformation on data points
    Transformation = np.linalg.inv(Transformation)
    image = np.dot(
        Transformation, np.concatenate((image.T, np.ones((1, image.shape[0]))))
    )

    # retrieve normalized image in the original input shape (25, 2)
    image = image[0:2].T

    # print("translated origin:", np.mean(image, axis=0))
    # print("average distance to origin:", get_average_dist_to_origin(image))

    return image, Transformation


def compute_matrix_A(points1, points2):
    """
    Input: Normalized correspondences for image1 and image2
    Output: Matrix A as defined in Zisserman p. 91
    """

    A = []
    for i in range(0, len(points1)):
        x, y = points1[i, 0], points1[i, 1]
        x_prime, y_prime = points2[i, 0], points2[i, 1]

        # create A_i according to the eq. in the book
        # here we are assuming w_i is one
        A.append([0, 0, 0, -x, -y, -1, y_prime * x, y_prime * y, y_prime])
        A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y, -x_prime])

    # print()
    # print("Stacked matrix A shape:", np.shape(A))

    return np.asarray(A)


def compute_SVD(matrix_A):
    # print()
    # print("Computing SVD...")

    return np.linalg.svd(matrix_A)


def get_vector_h(matrix_V):
    """
    Input: Matrix V from SVD of A
    Output: Unitary vector h (last column of V matrix of SVD)
    """
    # print()
    # print("Obtaining vector h...")

    h = matrix_V[-1, :] / matrix_V[-1, -1]
    return h


def calculate_homography(points1, points2, normalize=True):
    # set data points to numpy arrays
    points1 = np.array(points1)
    points2 = np.array(points2)

    # normalize data
    if normalize:
        points1, T = normalize_image_points(points1)
        points2, T_prime = normalize_image_points(points2)

    # get matrix A for each normalized correspondence (dims 2*n x 9)
    A = compute_matrix_A(points1, points2)

    # compute SVD of A
    U, S, V = compute_SVD(A)

    # get last column of V and normalize it (this is the vector h)
    h = get_vector_h(V)

    # obtain homography (H tilde)
    # print()
    # print("Reshaping to get homography H_tilde...")
    H = h.reshape(3, 3)

    if normalize:
        # denormalize to obtain homography (H) using the transformations and generalized pseudo-inverse
        H = np.dot(np.dot(np.linalg.pinv(T_prime), H), T)

    # print()
    # print("Denormalized to obtain homography H for 2D data points...")
    # print("Matrix H:")
    # print(H)
    return H


def calculate_error(points1, points2):
    # Sum the squared differences
    sum_squared_diff = np.sum((points1 - points2) ** 2)
    # Calculate the average distance
    average_distance = np.sqrt(sum_squared_diff) / len(points1)

    return average_distance


if __name__ == "__main__":
    img1Path = "images/1-0.png"
    img2Path = "images/1-1.png"
    img3Path = "images/1-2.png"

    gt_correspondences_1 = np.load("groundtruth_correspondences/correspondence_01.npy")
    gt_correspondences_2 = np.load("groundtruth_correspondences/correspondence_02.npy")

    for i, path in enumerate([img2Path, img3Path]):
        image1 = cv.imread(img1Path)
        image2 = cv.imread(path)
        points1, points2 = get_sift_correspondences(image1, image2, 50)
        if i == 1:  # for second image, remove bad matches
            points1 = np.concatenate((points1[:5], points1[6:14], points1[15:]), axis=0)
            points2 = np.concatenate((points2[:5], points2[6:14], points2[15:]), axis=0)
        np.save(f"sift_correspondences/correspondences_0{i+1}.npy", [points1, points2])

    for i in range(2):
        gt = np.load(f"groundtruth_correspondences/correspondence_0{i+1}.npy")
        for k in [4, 8, 20]:
            points1, points2 = np.load(
                f"sift_correspondences/correspondences_0{i+1}.npy"
            )
            points1, points2 = points1[:k], points2[:k]
            for normalize in [False, True]:
                H = calculate_homography(points1, points2, normalize=normalize)

                g_s, g_t = gt
                g_s = np.append(g_s, np.ones(100).reshape(100, 1), axis=1)
                pred = np.dot(H, g_s.transpose())
                pred = (pred[:2] / pred[2]).transpose()
                error = calculate_error(pred, g_t)
                print(
                    f"""Image: {path}\nNormalize: {normalize}\nNumber of sample points: {k}"""
                )
                print("Error:", error)
                print()
