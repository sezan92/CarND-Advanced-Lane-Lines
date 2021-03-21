import cv2
import matplotlib.pyplot as plt
import numpy as np


class Camera:
    def __init__(self):
        """Camera Class """
        self.obj_points = []
        self.img_points = []

    def set_img_points(self, img):
        """
        numpy.ndarray -> None
        Appends real world points and image points to self.obj_points
        and self.img_points respectively
        """
        obj_point = np.zeros((6 * 9, 3), np.float32)
        obj_point[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        corners = self.get_chessboard_corner(img)
        if corners is not None:
            self.obj_points.append(obj_point)
            self.img_points.append(corners)

    def get_calibration_matrix(self, img_size):
        """
        tuple -> numpy.ndarray
        Returns Calbration Matrix of given image_size
        parameters:
            img_size: (tuple), image shape
        returns:
            calibration_matrix: numpy array, calibration matrix
        """

        ret, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(
            self.obj_points, self.img_points, img_size, None, None
        )
        if not ret:
            cameraMatrix = None
            distCoeffs = None
        return cameraMatrix, distCoeffs

    def get_chessboard_corner(self, img, plot=False):
        """
        Returns chessboard corners
        parameters:
            img: numpy array, the image
            plot: boolean, to plot , or not to plot
        returns:
            object points: numpy array
            corners: #TODO
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            image = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            if plot:
                plt.figure()
                plt.imshow(image)

            return corners
        else:
            return None
