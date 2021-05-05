import unittest

from lane_utils import LaneDetector


class TestLaneDetector(unittest.TestCase):
    def test_init(self):
        ld = LaneDetector()
        assert ld.nwindows == 20
        assert ld.margin == 60
        assert ld.minpix == 25

    def test_set_nwindows(self):
        ld = LaneDetector()
        ld.set_nwindows(10)
        assert ld.nwindows == 10

    def test_set_margin(self):
        ld = LaneDetector()
        ld.set_margin(100)
        assert ld.margin == 100

    def test_minpix(self):
        ld = LaneDetector()
        ld.set_minpix(1000)
        assert ld.minpix == 1000


if __name__ == "__main__":
    unittest.main()
