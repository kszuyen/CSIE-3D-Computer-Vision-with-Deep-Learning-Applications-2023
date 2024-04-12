import pandas as pd
import numpy as np
import open3d as o3d


def visualization(camera_position_df, points3D_df):
    geometries_list = []
    for i in range(len(camera_position_df)):
        r = camera_position_df.iloc[i].rotation
        p = camera_position_df.iloc[i].position

        model = o3d.geometry.LineSet()
        model.points = o3d.utility.Vector3dVector(
            [[0, 0, 0], [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]]
        )
        model.lines = o3d.utility.Vector2iVector(
            [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        )
        color = np.array([1, 0, 0])
        model.colors = o3d.utility.Vector3dVector(np.tile(color, (8, 1)))
        model.scale(0.03, np.zeros(3))
        model.rotate(r)
        model.translate(p)
        geometries_list.append(model)
    points3D_positions = np.array(points3D_df["XYZ"].to_list())
    points3D_colors = np.array(points3D_df["RGB"].to_list()) / 255

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points3D_positions)
    point_cloud.colors = o3d.utility.Vector3dVector(points3D_colors)
    geometries_list.append(point_cloud)

    o3d.visualization.draw_geometries(
        geometries_list,
        zoom=0.06,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[1.6172, 0.3475, 0.132],
        up=[-0.0694, -0.9768, 0.2024],
    )
    # o3d.visualization.draw_geometries([line_set, mesh, point_cloud])


def main():
    points3D_df = pd.read_pickle("data/points3D.pkl")
    camera_position_df = pd.read_pickle("./camera_position.pkl")
    visualization(camera_position_df, points3D_df)


if __name__ == "__main__":
    main()
