import unittest

import util
from preprocess import PerspectiveTransformer


class TestPerspectiveTransformer(unittest.TestCase):
    def test_set_config(self):
        transformer = PerspectiveTransformer()
        img = util.imread("test_images/test2.jpg")
        transformer.set_config(img)
        assert len(transformer.pts) == 4


if __name__ == "__main__":
    unittest.main()
