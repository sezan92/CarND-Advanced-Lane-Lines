import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from util import load_img_dir


class Camera:
    def __init__(self, img_size=(None, None)):
        """Camera Class """
        self.obj_points = []
        self.img_points = []
        self.img_size = img_size
        self.calib_matrix = None
        self.dist_coeffs = None

    def set_img_points(self, img, plot="False", fig_name=None):
        """
        numpy.ndarray -> None
        Appends real world points and image points to self.obj_points
        and self.img_points respectively
        parameters:
            img: np.ndarray, image to get the chessboard corners
            plot: bool, to plot, or not to plot
            fig_name: str, figure name
        """
        obj_point = np.zeros((6 * 9, 3), np.float32)
        obj_point[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        corners = self._get_chessboard_corner(img, plot, fig_name)
        if corners is not None:
            self.obj_points.append(obj_point)
            self.img_points.append(corners)

    def get_calibration_matrix(self):
        """

        Returns Calbration Matrix of a given camera
        returns:
            calibration_matrix: numpy array, calibration matrix
            dist_coeffs: numpy array, dist coeffecients
        """
        return self.calib_matrix, self.dist_coeffs

    def tune(self, img_dir, cfg_filename="camera.yaml", camera_cal_output_dir=None):
        """
        Tunes camera configuration based on the given image dir
        parameters:
            img_dir: str, directory of images intended to be used
            cfg_filename: str, confirutration filename

        """
        imgs = load_img_dir(img_dir)

        for i, img in enumerate(imgs):
            if camera_cal_output_dir is not None:
                fig_name = os.path.join(camera_cal_output_dir, f"{i}.jpg")
            else:
                fig_name = None
            self.set_img_points(
                img,
                plot=True,
                fig_name=fig_name,
            )
        if len(self.obj_points) == 0:
            raise "ERROR! Please add chessboard " + " image points using set_img_points method"

        ret, self.calib_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(
            self.obj_points, self.img_points, self.img_size, None, None
        )
        if not ret:
            self.calib_matrix = None
            self.dist_coeffs = None
        cfg = {
            "calib_matrix": self.calib_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
        }
        with open(cfg_filename, "w") as cfgh:
            cfgh.dump(cfg, cfgh)
            logging.info(f"saved calibration matrix in {cfg_filename}")

    def _get_chessboard_corner(self, img, plot=False, fig_name=None):
        """
        Returns chessboard corners
        parameters:
            img: numpy array, the image
            plot: boolean, to plot , or not to plot
        returns:
            corners: numpy array, the corners
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            image = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            if plot:
                plt.figure()
                plt.imshow(image)
                if fig_name is not None:
                    plt.savefig(fig_name)

            return corners
        else:
            return None

    def transform(self, img):
        """
        numpy.ndarray -> numpy.ndarray
        Returns undistorted image of a given image , using cameras
        intrinsic properties
        Parameters:
            img: numpy.ndarray, Input img
        Returns:
            undistort: numpy.ndarray, undistorted image
        """
        if self.calib_matrix is None:
            raise "Error! No calibration matrix!" + "Please calibrate the camera using the checkboard images!"
        undistort = cv2.undistort(
            img, self.calib_matrix, self.dist_coeffs, None, self.calib_matrix
        )

        return undistort
