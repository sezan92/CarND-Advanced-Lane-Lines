import unittest

import util
from calibration import get_chessboard_corner


class TestCalibration(unittest.TestCase):
    def test_get_chessboard_corner1(self):
        img = util.imread("camera_cal/calibration1.jpg")
        object_point, corners = get_chessboard_corner(img, False)

        assert object_point is None
        assert corners is None

    def test_get_chessboard_corner2(self):
        img = util.imread("camera_cal/calibration2.jpg")
        object_point, corners = get_chessboard_corner(img, False)

        assert object_point is not None
        assert corners is not None


if __name__ == "__main__":
    unittest.main()
