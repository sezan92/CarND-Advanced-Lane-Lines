import cv2
import numpy as np


class Transformer:
    def __init__(self):
        """
        Base class for transofrmer
        """
        pass

    def tune(self, img):
        pass

    def transform(self, img):
        pass


class PerspectiveTransformer:
    def __init__(self):
        self.pts = []
        self.threshold = 400

    def tune(self, img):
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

    def r2b(self, img):
        """
        Converts RGB to Binary image
        Arguments:
            img: numpy array, input image
            s1: int, s channel lowest threshold
            s2: int, s channel highest threshold
            sx1: int, sobel operator x axis lowest threshold
            sx2: int, sobel operator x axis highest Threshold
        Returns:
            combined_binary: numpy array, binary image
        """
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[
            (scaled_sobel >= self.sx_thresh[0]) & (scaled_sobel <= self.sx_thresh[1])
        ] = 1

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 1
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return combined_binary

    def tune(self, img):
        """
        tunes a given image
        parameters:
            img: numpy.ndarray, the given image to tune
        returns:
            None

        """
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
            bw = self.r2b(img)
            cv2.imshow(self.tuning_title, bw * 255)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break

    def _get_mean_of_threshold(self, threshold_list):
        """
        Gets the mean of threshold list
        Arguments:
            threshold_list: list, list of tuples
        Returns:
            mean_value: tuple, mean of the tuples
        """
        sum1 = 0
        sum2 = 0
        for (t1, t2) in threshold_list:
            sum1 = sum1 + t1
            sum2 = sum2 + t2
        mean_value = (sum1 / len(threshold_list), sum2 / len(threshold_list))

        return mean_value

    def tune_imgs(self, imgs):
        """
        Tunes a given directory and saves the average threshold values
        Arguments:
            imgs: list, list of images

        """
        s_values = []
        sx_values = []
        for img in imgs:
            self.tune(img)
            s_values.append(self.s_thresh)
            sx_values.append(self.sx_thresh)

        self.s_thresh = self._get_mean_of_threshold(s_values)
        self.sx_thresh = self._get_mean_of_threshold(sx_values)

    def transform(self, img):

        return self.r2b(img)
