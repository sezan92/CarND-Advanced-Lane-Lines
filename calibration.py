import cv2
import matplotlib.pyplot as plt
import numpy as np
from util import imread

def get_chessboard_corner(image, plot=False):
    """
    Returns chessboard corners
    parameters:
        image: numpy array, the image
        plot: boolean, to plot , or not to plot
    returns:
        object points: numpy array
        corners: #TODO
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    object_point = np.zeros((6 * 9, 3), np.float32)
    object_point[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    
    if ret:
        image = cv2.drawChessboardCorners(image, (9, 6), corners, ret)
        if plot:
            plt.figure()
            plt.imshow(image)

        return object_point, corners
    else:
        return None, None

    

    