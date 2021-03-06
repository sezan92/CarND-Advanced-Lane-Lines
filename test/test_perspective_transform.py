import os
import unittest

import cv2
import numpy as np

import util
from preprocess import PerspectiveTransformer


class TestPerspectiveTransformer(unittest.TestCase):
    def test_tune(self):
        self.transformer = PerspectiveTransformer()
        img = util.imread("test_images/test2.jpg", "BGR")
        self.transformer.tune(img)
        assert len(self.transformer.pts) == 4
        assert isinstance(self.transformer.M, np.ndarray)
        assert isinstance(self.transformer.inverse_M, np.ndarray)
        img = util.imread("test_images/test1.jpg", "RGB")
        img_transformed = self.transformer.transform(img)
        cv2.imshow("transformed", img_transformed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img_inverse_transformed = self.transformer.inverse_transform(img_transformed)
        cv2.imshow("inverse transformd", img_inverse_transformed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        assert isinstance(img_transformed, np.ndarray)
        assert os.path.exists("config.npy")
        self.transformer.load_config("config.npy")
        assert isinstance(self.transformer.M, np.ndarray)


if __name__ == "__main__":
    unittest.main()
