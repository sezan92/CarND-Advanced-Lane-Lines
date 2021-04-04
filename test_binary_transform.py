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


if __name__ == "__main__":
    unittest.main()
