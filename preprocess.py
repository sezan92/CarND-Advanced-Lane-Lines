import cv2
import numpy as np


class PerspectiveTransformer:
    def __init__(self):
        self.pts = []
        self.threshold = 400

    def set_config(self, img):
        """
        Sets configuration for Perspective Transform
        Args:
            img: numpy.array , img for setting configuration

        """
        self.img = img
        cv2.imshow("original", self.img)
        cv2.setMouseCallback("original", self._click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.H, self.W, _ = self.img.shape
        src = np.float32(self.pts)
        dest = np.float32(
            [
                (self.threshold, 0),  # top left
                (self.W - self.threshold, 0),  # top right
                (self.threshold, self.H),  # bottom left
                (self.W - self.threshold, self.H),
            ]
        )  # bottom right
        self.M = cv2.getPerspectiveTransform(src, dest)
        # TODO: sort the self.pts perfectly to destination

    def transform(self, img):
        """
        Perspective Transformation
        Ars:
            img: numpy.array, image input
        Returns:
            img: numpy.array, transformed image
        """
        img = cv2.warpPerspective(img, self.M, (self.W, self.H), flags=cv2.INTER_LINEAR)

        return img

    def _click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pts) < 4:
                self.pts.append((x, y))
                self.img = cv2.circle(
                    self.img, (x, y), radius=5, color=(0, 0, 255), thickness=-1
                )
                cv2.imshow("original", self.img)
            else:
                print("got all points!")
                print(self.pts)

class BinaryTransformer:
    def __init__(self, s=(0, 255), sx=(0, 255)):
        self.s = s
        self.sx = sx
    
    def tune(self, img, s1, s2, sx1, sx2, plot=False):
        pass

    def transform(self, img):
        pass

