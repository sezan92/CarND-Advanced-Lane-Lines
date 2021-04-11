from __future__ import unicode_literals

import os
import unittest

import numpy as np
import yaml

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
        self.transformer.tune_imgs(imgs)
        assert isinstance(self.transformer.s_thresh, tuple)
        assert isinstance(self.transformer.sx_thresh, tuple)
        assert os.path.exists("config.yaml")
        with open("config.yaml", "r") as cfgh:
            cfg = yaml.load(cfgh)
        assert isinstance(cfg, dict)
        assert "s_thresh" in cfg
        assert "sx_thresh" in cfg


if __name__ == "__main__":
    unittest.main()
