from functools import partial

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
    def __init__(self, s_thresh=(None, None), sx_thresh=(None, None)):
        self.s_thresh = s_thresh
        self.sx_thresh = sx_thresh
        self.tuning_title = "Tuning"
        self.trackbar_name = "b&w track"

    def __nothing__(self, val):
        pass

    def r2b(self, img, s1=150, s2=230, sx1=89, sx2=220):
        s_thresh = (s1, s2)
        sx_thresh = (sx1, sx2)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return combined_binary

    def tune(self, img):

        cv2.namedWindow(self.tuning_title)
        cv2.createTrackbar("s1", self.tuning_title, 0, 255, self.__nothing__)
        cv2.createTrackbar("s2", self.tuning_title, 0, 255, self.__nothing__)
        cv2.createTrackbar("sx1", self.tuning_title, 0, 255, self.__nothing__)
        cv2.createTrackbar("sx2", self.tuning_title, 0, 255, self.__nothing__)

        while True:
            s1 = cv2.getTrackbarPos("s1", self.tuning_title)
            s2 = cv2.getTrackbarPos("s2", self.tuning_title)
            sx1 = cv2.getTrackbarPos("sx1", self.tuning_title)
            sx2 = cv2.getTrackbarPos("sx2", self.tuning_title)
            self.s_thresh = (s1, s2)
            self.sx_thresh = (sx1, sx2)
            bw = self.r2b(img, s1, s2, sx1, sx2)
            cv2.imshow(self.tuning_title, bw * 255)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break

    def transform(self, img):
        pass
