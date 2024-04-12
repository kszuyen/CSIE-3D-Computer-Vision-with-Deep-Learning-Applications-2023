import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp


class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params["K"]
        self.dist = camera_params["dist"]

        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, "*.png"))))

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue,))
        p.start()

        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    # TODO:
                    # insert new camera pose here using vis.add_geometry()
                    # polyhedron shape:
                    pyramid = o3d.geometry.LineSet()
                    pyramid.points = o3d.utility.Vector3dVector(
                        [[0, 0, 0], [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]]
                    )
                    pyramid.lines = o3d.utility.Vector2iVector(
                        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
                    )
                    color = np.array([1, 0, 0])
                    pyramid.colors = o3d.utility.Vector3dVector(np.tile(color, (8, 1)))
                    pyramid.rotate(R)
                    pyramid.translate(t)
                    vis.add_geometry(pyramid)
            except:
                pass

            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def process_frames(self, queue):
        prev_R, prev_t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        # read 1st image
        img1 = self.read_image(self.frame_paths[0])

        for frame_path in self.frame_paths[1:]:
            # img = cv.imread(frame_path)
            # TODO: compute camera pose here
            img2 = self.read_image(frame_path)
            points1, points2 = self.extract_feature(img1, img2)

            # find essential matrix
            E, mask = cv.findEssentialMat(points1, points2, self.K, threshold=1)
            # recover pose
            retval, R, t, mask = cv.recoverPose(E, points1, points2, self.K, mask=mask)
            # calculate relative pose
            R = R.dot(prev_R)
            t = prev_t + R.dot(t)

            queue.put((R, t))

            # Draw the matched (tracked) point on current image
            for point in points2:
                cv.circle(img2, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)

            cv.imshow("frame", img2)

            # update img1, prev_R, prev_t
            img1 = img2
            prev_R = R
            prev_t = t

            if cv.waitKey(30) == 27:
                break

    def read_image(self, image_path):
        img = cv.imread(image_path)
        img = cv.undistort(img, self.K, self.dist)
        return img

    def extract_feature(self, img1, img2):
        """extract features with orb feature extractor"""
        # Initiate ORB detector
        orb = cv.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        points1 = np.array([kp1[m.queryIdx].pt for m in matches])
        points2 = np.array([kp2[m.trainIdx].pt for m in matches])

        return points1, points2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="directory of sequential frames")
    parser.add_argument(
        "--camera_parameters",
        default="camera_parameters.npy",
        help="npy file of camera parameters",
    )
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
