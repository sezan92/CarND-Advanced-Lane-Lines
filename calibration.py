import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from util import imread


def get_chessboard_corner(img, plot=False):
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


def get_calibration_matrix(folder_name):
    """
    Returns Calbration Matrix of the chessboard image
    parameters:
        folder_name: str, path to images
    returns:
        calibration_matrix: numpy array, calibration matrix
    """
    obj_point = np.zeros((6 * 9, 3), np.float32)
    obj_point[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    images = [
        os.path.join(folder_name, image_path)
        for image_path in os.listdir(folder_name)
        if image_path.endswith("jpg")
    ]

    obj_points = []
    img_points = []

    for image_name in images:
        img = imread(image_name)
        corners = get_chessboard_corner(img)
        if corners is not None:
            obj_points.append(obj_point)
            img_points.append(corners)

    img_size = img.shape[:2]
    ret, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    return cameraMatrix, distCoeffs
