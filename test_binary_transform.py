from __future__ import unicode_literals

import unittest

import numpy as np

import util
from preprocess import BinaryTransformer


class TestBinaryTransformer(unittest.TestCase):
    def test_tune(self):
        self.transformer = BinaryTransformer()
        img = util.imread("test_images/test2.jpg", "BGR")
        bw = self.transformer.tune(img)
        assert isinstance(bw, np.ndarray)

    def test_tune_dir(self):
        self.transformer = BinaryTransformer()
        dir = "test_images"
        imgs = util.load_img_from_dir(dir)
        self.transformer.tune_dir(imgs)
        assert isinstance(self.transformer.s_thresh, tuple)
        assert isinstance(self.sx_thresh, tuple)


if __name__ == "__main__":
    unittest.main()
