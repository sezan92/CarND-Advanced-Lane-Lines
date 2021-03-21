import unittest

import numpy

import util
from calibration import Camera


class TestCalibration(unittest.TestCase):
    def test_get_chessboard_corner1(self):
        camera = Camera((720, 1280))

        img = util.imread("camera_cal/calibration1.jpg")
        corners = camera._get_chessboard_corner(img, False)

        assert corners is None

    def test_get_chessboard_corner2(self):
        camera = Camera((720, 1280))
        img = util.imread("camera_cal/calibration2.jpg")
        corners = camera._get_chessboard_corner(img, False)

        assert corners is not None

    def test_set_img_points1(self):
        camera = Camera((720, 1280))
        img = util.imread("camera_cal/calibration1.jpg")
        camera.set_img_points(img)

        assert len(camera.obj_points) == 0
        assert len(camera.img_points) == 0

    def test_set_img_points2(self):
        camera = Camera((720, 1280))
        img = util.imread("camera_cal/calibration2.jpg")
        camera.set_img_points(img)
        assert len(camera.obj_points) == 1
        assert len(camera.img_points) == 1

    def test_get_calibration_matrix(self):
        camera = Camera((720, 1280))
        img = util.imread("camera_cal/calibration2.jpg")
        camera.set_img_points(img)
        cameraMatrix, distCoeffs = camera.get_calibration_matrix()
        assert isinstance(cameraMatrix, numpy.ndarray)


if __name__ == "__main__":
    unittest.main()
