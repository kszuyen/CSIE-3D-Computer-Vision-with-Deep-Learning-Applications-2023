import open3d as o3d
import cv2
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
from tqdm import tqdm

PNPSOLVER = __import__("2d3dmatching").pnpsolver

cameraMatrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
distCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352])

print("loading data...")
images_df = pd.read_pickle("data/images.pkl")
train_df = pd.read_pickle("data/train.pkl")
points3D_df = pd.read_pickle("data/points3D.pkl")
point_desc_df = pd.read_pickle("data/point_desc.pkl")


def load_point_cloud(points3D_df):
    xyz = np.vstack(points3D_df["XYZ"])
    rgb = np.vstack(points3D_df["RGB"]) / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


def load_axes():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    axes.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])  # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )  # R, G, B
    return axes


def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler("xyz", rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate(
        [scale_mat @ r_mat, translation.reshape(3, 1)], axis=1
    )
    return transform_mat


def update_cube():
    global cube, cube_vertices, R_euler, t, scale

    transform_mat = get_transform_mat(R_euler, t, scale)

    transform_vertices = (
        transform_mat
        @ np.concatenate(
            [cube_vertices.transpose(), np.ones([1, cube_vertices.shape[0]])], axis=0
        )
    ).transpose()

    cube.vertices = o3d.utility.Vector3dVector(transform_vertices)
    cube.compute_vertex_normals()
    cube.paint_uniform_color([1, 0.706, 0])
    vis.update_geometry(cube)


def toggle_key_shift(vis, action, mods):
    global shift_pressed
    if action == 1:  # key down
        shift_pressed = True
    elif action == 0:  # key up
        shift_pressed = False
    return True


def update_tx(vis):
    global t, shift_pressed
    t[0] += -0.01 if shift_pressed else 0.01
    update_cube()


def update_ty(vis):
    global t, shift_pressed
    t[1] += -0.01 if shift_pressed else 0.01
    update_cube()


def update_tz(vis):
    global t, shift_pressed
    t[2] += -0.01 if shift_pressed else 0.01
    update_cube()


def update_rx(vis):
    global R_euler, shift_pressed
    R_euler[0] += -1 if shift_pressed else 1
    update_cube()


def update_ry(vis):
    global R_euler, shift_pressed
    R_euler[1] += -1 if shift_pressed else 1
    update_cube()


def update_rz(vis):
    global R_euler, shift_pressed
    R_euler[2] += -1 if shift_pressed else 1
    update_cube()


def update_scale(vis):
    global scale, shift_pressed
    scale += -0.05 if shift_pressed else 0.05
    update_cube()


def average(x):
    return list(np.mean(x, axis=0))


def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID", "XYZ", "RGB", "DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc


def Get_Image_R_T(method):
    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    global images_df

    # Preprocess images_df
    images_df = images_df[images_df.NAME.str.startswith("valid")]
    new_id_list = []
    for name in images_df.NAME:
        new_name = name[9:]
        new_name = new_name[:-4]
        new_id_list.append(int(new_name))
    images_df["NEW_ID"] = new_id_list
    images_df.sort_values(by=["NEW_ID"], inplace=True)

    R_list, T_list, img_list = [], [], []

    for idx in tqdm(images_df.IMAGE_ID):
        # Load query image
        fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
        rimg = cv2.imread("data/frames/" + fname)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        # rvec, tvec, inliers = PNPSOLVER((kp_query, desc_query), (kp_model, desc_model))
        rotq, tvec = PNPSOLVER(
            (kp_query, desc_query),
            (kp_model, desc_model),
            cameraMatrix,
            distCoeffs,
            method,
        )

        # Find correspondance and solve pnp
        rotq, tvec = PNPSOLVER(
            (kp_query, desc_query),
            (kp_model, desc_model),
            cameraMatrix,
            distCoeffs,
            method,
        )
        r = R.from_quat(rotq).as_matrix().reshape(3, 3)
        t = tvec.reshape(1, 3)

        img_list.append(rimg)
        R_list.append(r)
        T_list.append(t)

    img_list, R_list, T_list = (
        np.array(img_list),
        np.array(R_list).reshape(-1, 3, 3),
        np.array(T_list).reshape(-1, 1, 3),
    )
    return img_list, R_list, T_list


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# load point cloud
points3D_df = pd.read_pickle("data/points3D.pkl")
pcd = load_point_cloud(points3D_df)
vis.add_geometry(pcd)

# load axes
axes = load_axes()
vis.add_geometry(axes)

# load cube
cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
cube_vertices = np.asarray(cube.vertices).copy()
vis.add_geometry(cube)

R_euler = np.array([0, 0, 0]).astype(float)
t = np.array([0, 0, 0]).astype(float)
scale = 1.0
update_cube()

# just set a proper initial camera view
vc = vis.get_view_control()
vc_cam = vc.convert_to_pinhole_camera_parameters()
initial_cam = get_transform_mat(
    np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1
)
initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
initial_cam[-1, -1] = 1.0
setattr(vc_cam, "extrinsic", initial_cam)
vc.convert_from_pinhole_camera_parameters(vc_cam)

# set key callback
shift_pressed = False
vis.register_key_action_callback(340, toggle_key_shift)
vis.register_key_action_callback(344, toggle_key_shift)
vis.register_key_callback(ord("A"), update_tx)
vis.register_key_callback(ord("S"), update_ty)
vis.register_key_callback(ord("D"), update_tz)
vis.register_key_callback(ord("Z"), update_rx)
vis.register_key_callback(ord("X"), update_ry)
vis.register_key_callback(ord("C"), update_rz)
vis.register_key_callback(ord("V"), update_scale)

print("[Keyboard usage]")
print("Translate along X-axis\tA / Shift+A")
print("Translate along Y-axis\tS / Shift+S")
print("Translate along Z-axis\tD / Shift+D")
print("Rotate    along X-axis\tZ / Shift+Z")
print("Rotate    along Y-axis\tX / Shift+X")
print("Rotate    along Z-axis\tC / Shift+C")
print("Scale                 \tV / Shift+V")

vis.run()
vis.destroy_window()

"""
print('Rotation matrix:\n{}'.format(R.from_euler('xyz', R_euler, degrees=True).as_matrix()))
print('Translation vector:\n{}'.format(t))
print('Scale factor: {}'.format(scale))
"""

np.save("cube_transform_mat.npy", get_transform_mat(R_euler, t, scale))
np.save("cube_vertices.npy", np.asarray(cube.vertices))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ["p3p", "dlt"]:
        print("Usage: \npython transform_cube.py <method: [p3p]or[dlt]>")
        sys.exit()
    else:
        method = sys.argv[1]

    all_points, point_colors = np.empty((0, 3), float), np.empty((0, 3), float)

    # n points on one side
    n = 10

    cube_corner = np.load("cube_vertices.npy")

    origin, diagonal = cube_corner[0, :], cube_corner[7, :]
    all_points = np.append(all_points, np.array(origin).reshape(1, 3), axis=0)
    point_colors = np.append(point_colors, np.array([0, 0, 255]).reshape(1, 3), axis=0)

    axis1, axis2, axis3 = (
        cube_corner[1, :] - origin,
        cube_corner[2, :] - origin,
        cube_corner[4, :] - origin,
    )

    origin_planes = [[axis1, axis2], [axis1, axis3], [axis2, axis3]]
    origin_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    diagonal_planes = [[-axis1, -axis2], [-axis1, -axis3], [-axis2, -axis3]]
    diagonal_colors = [[255, 255, 0], [255, 0, 255], [0, 255, 255]]

    ori = [origin, origin_planes, origin_colors]
    dia = [diagonal, diagonal_planes, diagonal_colors]

    # Interpolation points
    for base, planes, colors in [ori, dia]:
        for [axis, axis_], color in zip(planes, colors):
            for x in range(n):
                for y in range(n):
                    all_points = np.append(
                        all_points,
                        (base + axis * (x / n) + axis_ * (y / n)).reshape(1, 3),
                        axis=0,
                    )
                    point_colors = np.append(
                        point_colors, np.array(color).reshape(1, 3), axis=0
                    )

    K = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
    img_list, R_list, T_list = Get_Image_R_T(method)

    videowrt = cv2.VideoWriter(
        f"AR_video_{method}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (1080, 1920)
    )

    for img, r, t in zip(img_list, R_list, T_list):
        M = K.dot(np.concatenate((r, t.T), axis=1))

        for point, color in zip(all_points, point_colors):
            point2D = M.dot(np.append(point, 1))
            point2D = point2D[:-1] / point2D[-1]
            point2D = np.int_(point2D)
            if (
                point2D[0] >= 0
                and point2D[0] < 1080
                and point2D[1] >= 0
                and point2D[1] < 1920
            ):
                img = cv2.circle(img, point2D, 5, color, thickness=-1)
        videowrt.write(img)
    videowrt.release()


if __name__ == "__main__":
    main()
