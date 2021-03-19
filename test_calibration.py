import unittest

import numpy

import util
from calibration import get_calibration_matrix, get_chessboard_corner


class TestCalibration(unittest.TestCase):
    def test_get_chessboard_corner1(self):
        img = util.imread("camera_cal/calibration1.jpg")
        corners = get_chessboard_corner(img, False)

        assert corners is None

    def test_get_chessboard_corner2(self):
        img = util.imread("camera_cal/calibration2.jpg")
        corners = get_chessboard_corner(img, False)

        assert corners is not None

    def test_get_calibration_matrix(self):
        cameraMatrix, distCoeffs = get_calibration_matrix("camera_cal")
        assert isinstance(cameraMatrix, numpy.ndarray)


if __name__ == "__main__":
    unittest.main()
