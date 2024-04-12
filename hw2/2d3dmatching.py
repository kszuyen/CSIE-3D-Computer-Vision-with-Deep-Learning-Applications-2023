from scipy.spatial.transform import Rotation
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polycompanion
from numpy.linalg import eig, norm, inv, det, svd
from scipy.spatial import distance
from cv2 import BFMatcher, undistortImagePoints
import random
from tqdm import trange
import sys


def read_data():
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")
    return images_df, train_df, points3D_df, point_desc_df


# from the hw2 slides
cameraMatrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
distCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352])


def average(x):
    return list(np.mean(x, axis=0))


def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID", "XYZ", "RGB", "DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc


def pnpsolver(query, model, cameraMatrix, distCoeffs, method):
    # first find matches
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = BFMatcher()
    matches = bf.knnMatch(desc_query, desc_model, k=2)

    gmatches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            gmatches.append(m)

    points2D = np.empty((0, 2))
    points3D = np.empty((0, 3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D, kp_query[query_idx]))
        points3D = np.vstack((points3D, kp_model[model_idx]))

    # then perform p3p algorithm with ransac
    # _, r, t, _ = cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)
    R, T = Ransac(points3D, points2D, cameraMatrix, distCoeffs, method)
    return R, T


def trilateration(P1, P2, P3, r1, r2, r3):
    p1 = np.array([0, 0, 0])
    p2 = np.array([P2[0] - P1[0], P2[1] - P1[1], P2[2] - P1[2]])
    p3 = np.array([P3[0] - P1[0], P3[1] - P1[1], P3[2] - P1[2]])

    v1 = p2 - p1
    v2 = p3 - p1

    Xn = v1 / norm(v1)

    tmp = np.cross(v1, v2)

    Zn = tmp / norm(tmp)

    Yn = np.cross(Xn, Zn)

    i = np.dot(Xn, v2)
    d = np.dot(Xn, v1)
    j = np.dot(Yn, v2)

    X = ((r1**2) - (r2**2) + (d**2)) / (2 * d)
    Y = (((r1**2) - (r3**2) + (i**2) + (j**2)) / (2 * j)) - ((i / j) * (X))
    Z1 = np.sqrt(max(0, r1**2 - X**2 - Y**2))
    Z2 = -Z1

    K1 = P1 + X * Xn + Y * Yn + Z1 * Zn
    K2 = p1 + X * Xn + Y * Yn - Z2 * Zn

    return K1, K2


def solveP3P(points3D, points2D, cameraMatrix, distCoeffs):
    # p3p from text book
    x1, x2, x3 = points3D[0], points3D[1], points3D[2]
    X = [x1, x2, x3]

    # deal with distortion
    points2D = undistortImagePoints(points2D, cameraMatrix, distCoeffs).reshape(3, 2)
    u = np.concatenate((points2D.T, np.ones((1, points2D.shape[0]))))
    v = inv(cameraMatrix).dot(u)

    v1, v2, v3 = v.T[0], v.T[1], v.T[2]
    V = [v1, v2, v3]

    C_ab, C_ac, C_bc = (
        distance.cosine(v1, v2),
        distance.cosine(v1, v3),
        distance.cosine(v2, v3),
    )
    R_ab, R_ac, R_bc = (
        distance.euclidean(x1, x2),
        distance.euclidean(x1, x3),
        distance.euclidean(x2, x3),
    )

    K1, K2 = (R_bc / R_ac) ** 2, (R_bc / R_ab) ** 2

    G4 = (K1 * K2 - K1 - K2) ** 2 - 4 * K1 * K2 * C_bc**2
    G3 = 4 * (K1 * K2 - K1 - K2) * K2 * (1 - K1) * C_ab + 4 * K1 * C_bc * (
        (K1 * K2 - K1 + K2) * C_ac + 2 * K2 * C_ab * C_bc
    )
    G2 = (
        (2 * K2 * (1 - K1) * C_ab) ** 2
        + 2 * (K1 * K2 - K1 - K2) * (K1 * K2 + K1 - K2)
        + 4
        * K1
        * (
            (K1 - K2) * C_bc**2
            + K1 * (1 - K2) * C_ac**2
            - 2 * (1 + K1) * K2 * C_ab * C_ac * C_bc
        )
    )
    G1 = 4 * (K1 * K2 + K1 - K2) * K2 * (1 - K1) * C_ab + 4 * K1 * (
        (K1 * K2 - K1 + K2) * C_ac * C_bc + 2 * K1 * K2 * C_ab * C_ac**2
    )
    G0 = (K1 * K2 + K1 - K2) ** 2 - 4 * K1**2 * K2 * C_ac**2
    companion_matrix = polycompanion([G0, G1, G2, G3, G4])
    roots = eig(companion_matrix)[0]
    R_list, T_list = [], []
    for root in roots:
        if isinstance(root, complex):
            continue
        a = ((R_ab**2) / (root**2 - 2 * root * C_ab + 1)) ** 0.5
        a_list = [a, -a]
        m, p, q = 1 - K1, 2 * (K1 * C_ac - root * C_bc), root**2 - K1
        m_prime, p_prime, q_prime = (
            1,
            2 * (-root * C_bc),
            (root**2 * (1 - K2) + 2 * root * K2 * C_ab - K2),
        )

        for a in a_list:
            y = -(m_prime * q - m * q_prime) / (p * m_prime - p_prime * m)
            # y = -(m_prime * q - m * q_prime) / (p_prime * q - p * q_prime)

            b = root * a
            c = y * a

            T = trilateration(x1, x2, x3, a, b, c)

            for t in T:
                lambda_v_list, xt_list = np.array([]), np.array([])

                for i in range(len(X)):
                    lambda_ = norm(X[i] - t) / norm(V[i])
                    xt = X[i] - t
                    lambda_v_list = np.append(lambda_v_list, lambda_ * V[i])
                    xt_list = np.append(xt_list, xt)

                lambda_v = lambda_v_list.reshape((3, 3)).T
                xt = xt_list.reshape((3, 3)).T
                R = lambda_v.dot(inv(xt))
                if det(R) > 0:
                    R_list.append(R)
                    T_list.append(t)

    return np.array(R_list), np.array(T_list)


def solveDLT(points3D, points2D, cameraMatrix, distCoeffs):
    # UndistortImagePoints
    points2D = undistortImagePoints(points2D, cameraMatrix, distCoeffs).reshape(
        points2D.shape[0], 2
    )

    A = []
    for i in range(len(points3D)):
        [X, Y, Z] = points3D[i]
        [u, v] = points2D[i]

        Cx, Cy = cameraMatrix[0, 2], cameraMatrix[1, 2]

        Cx_u = Cx - u
        Cy_v = Cy - v

        fx, fy = cameraMatrix[0, 0], cameraMatrix[1, 1]

        A.append(
            [X * fx, Y * fx, Z * fx, fx, 0, 0, 0, 0, X * Cx_u, Y * Cx_u, Z * Cx_u, Cx_u]
        )
        A.append(
            [0, 0, 0, 0, X * fy, Y * fy, Z * fy, fy, X * Cy_v, Y * Cy_v, Z * Cy_v, Cy_v]
        )

    U, S, VH_tmp = svd(A)
    R_tmp = np.array(
        [
            [VH_tmp[-1, 0], VH_tmp[-1, 1], VH_tmp[-1, 2]],
            [VH_tmp[-1, 4], VH_tmp[-1, 5], VH_tmp[-1, 6]],
            [VH_tmp[-1, 8], VH_tmp[-1, 9], VH_tmp[-1, 10]],
        ]
    )
    U, S, VH = np.linalg.svd(R_tmp)
    c = 1 / (np.sum(S) / 3)
    R = U.dot(VH)
    tmp = c * (
        X * VH_tmp[-1, 8] + Y * VH_tmp[-1, 9] + Z * VH_tmp[-1, 10] + VH_tmp[-1, 11]
    )
    if (tmp) < 0:
        c = -c
        R = -R
    t = c * np.array([VH_tmp[-1, 3], VH_tmp[-1, 7], VH_tmp[-1, 11]]).reshape(3, 1)
    return R.reshape(3, 3), t.reshape(
        3,
    )


def calculate_inlier(R, T, points3D, points2D, cameraMatrix, distCoeffs, threshold):
    num_points = len(points2D)
    points2D = undistortImagePoints(points2D, cameraMatrix, distCoeffs).reshape(
        num_points, 2
    )

    M = cameraMatrix.dot(np.concatenate((R, T[:, np.newaxis]), axis=1))
    proj_points = M.dot(np.concatenate((points3D.T, np.ones((1, points3D.shape[0])))))
    proj_points = (proj_points / proj_points[2, :])[:2].T
    distances = np.linalg.norm(points2D - proj_points, axis=1)
    cnt = np.sum(distances <= threshold)

    return cnt


def Ransac(points3D, points2D, cameraMatrix, distCoeffs, method):
    N = 50  # N: iteration number
    threshold = 100  # threshold: ransac boundary
    p_num = 3 if method == "p3p" else 6
    num_best_inliners, best_R, best_T = 0, None, None

    for i in range(N):
        # for each iteration, random sample 3 points and perform p3p
        chosen_idx = random.sample(range(len(points3D)), p_num)
        chosen_points3D = np.array([points3D[i] for i in chosen_idx])
        chosen_points2D = np.array([points2D[i] for i in chosen_idx])

        if method == "p3p":
            try:
                R_list, t_list = solveP3P(
                    chosen_points3D, chosen_points2D, cameraMatrix, distCoeffs
                )
                # save the result of the most inlier
                for R, t in zip(R_list, t_list):
                    inl = calculate_inlier(
                        R, t, points3D, points2D, cameraMatrix, distCoeffs, threshold
                    )
                    if inl > num_best_inliners:
                        num_best_inliners = inl
                        best_R = R
                        best_T = t
            except:
                continue
        else:
            R, t = solveDLT(chosen_points3D, chosen_points2D, cameraMatrix, distCoeffs)
            inl = calculate_inlier(
                R, t, points3D, points2D, cameraMatrix, distCoeffs, threshold
            )
            if inl > num_best_inliners:
                num_best_inliners = inl
                best_R = R
                best_T = t

    # print("num_best_inliners", num_best_inliners)
    return Rotation.from_matrix(best_R).as_quat(), best_T


def differences(rotq, tvec, rotq_gt, tvec_gt):
    d_t = np.linalg.norm(tvec - tvec_gt, 2)
    nor_rotq = rotq / np.linalg.norm(rotq)
    nor_rotq_gt = rotq_gt / np.linalg.norm(rotq_gt)
    dif_r = np.clip(np.sum(nor_rotq * nor_rotq_gt), 0, 1)
    d_r = np.degrees(np.arccos(2 * dif_r * dif_r - 1))

    return d_r, d_t


def find_camera_position(rotq, tvec):
    # r_matrix = Rotation.from_rotvec(rvec.reshape(1, 3)).as_matrix().reshape(3, 3)
    r_matrix = Rotation.from_quat(rotq).as_matrix().reshape(3, 3)
    t_matrix = tvec.reshape(3, 1)
    R_T = np.concatenate((r_matrix, t_matrix), axis=1)
    tmp = np.array([[0, 0, 0, 1]])
    R_T = np.concatenate((R_T, tmp), axis=0)
    R_inverse = np.linalg.inv(R_T)
    R_matrix = R_inverse[:3, :3]
    T_matrix = R_inverse[:3, 3]
    return R_matrix, T_matrix


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ["p3p", "dlt"]:
        print("Usage: \npython 2d3dmatching.py <method: [p3p]or[dlt]>")
        sys.exit()
    else:
        method = sys.argv[1]

    print("loading data...")
    # read data
    images_df, train_df, points3D_df, point_desc_df = read_data()

    print("processing...")
    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)
    df = pd.DataFrame(columns=["rotation", "position"])
    differences_rotation = []
    differences_transition = []

    for idx in trange(164, 293):  # 293
        # Load query image
        # fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
        # rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        rotq, tvec = pnpsolver(
            (kp_query, desc_query),
            (kp_model, desc_model),
            cameraMatrix,
            distCoeffs,
            method,
        )
        rotation, position = find_camera_position(rotq, tvec)

        new_data = pd.DataFrame({"rotation": [rotation], "position": [position]})
        df = pd.concat([df, new_data], ignore_index=True)

        tvec = tvec.reshape(1, 3)

        # Get camera pose groudtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"] == idx]
        rotq_gt = ground_truth[["QX", "QY", "QZ", "QW"]].values
        tvec_gt = ground_truth[["TX", "TY", "TZ"]].values

        d_r, d_t = differences(rotq, tvec, rotq_gt, tvec_gt)

        differences_rotation.append(d_r)
        differences_transition.append(d_t)

    differences_rotation = np.array(differences_rotation)
    differences_transition = np.array(differences_transition)
    err_r = np.median(differences_rotation)
    err_t = np.median(differences_transition)

    print(f"pose error:{err_t}, rotation error:{err_r}")

    df.to_pickle("./camera_position.pkl")


if __name__ == "__main__":
    main()
