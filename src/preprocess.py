import logging

import cv2
import numpy as np
import yaml


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

    def load_config(self, cfg_name):
        pass


class PerspectiveTransformer(Transformer):
    def __init__(self, img_shape, threshold=400):
        super().__init__()
        self.pts = []
        self.threshold = threshold
        self.H, self.W = img_shape

    def tune(self, img, cfg_name="pt_config.yaml"):
        """
        Sets configuration for Perspective Transform
        Args:
            img: numpy.array , img for setting configuration
            cfg_name: str, name for saving the transform Matrix. default config.npy"

        """
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("original", self.img)
        cv2.setMouseCallback("original", self._click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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
        self.inverse_M = cv2.getPerspectiveTransform(dest, src)
        # TODO: sort the self.pts perfectly to destination
        cfg = {"M": self.M.tolist(), "inverse_M": self.inverse_M.tolist()}
        with open(cfg_name, "w") as cfgh:
            yaml.dump(cfg, cfgh)
        logging.info(f"saved configuration in {cfg_name}")

    def load_config(self, cfg_name="pt_config.yaml"):
        """
        method to load configuration.
        Arguments:
            cfg_name: str, configuration filename, default config.npy
        """
        with open(cfg_name, "r") as cfgh:
            cfg = yaml.load(cfgh)

            self.M = np.array(cfg["M"])
            self.inverse_M = np.array(cfg["inverse_M"])

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

    def inverse_transform(self, img_transformed):

        """Inverse transformation.
        Arguments:
            img: numpy.array, image input
        Returns:
            img: numpy.array, inverse transformed
        """

        img = cv2.warpPerspective(
            img_transformed, self.inverse_M, (self.W, self.H), flags=cv2.INTER_LINEAR
        )

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


class BinaryTransformer(Transformer):
    def __init__(self, s_thresh=(None, None), sobel_thresh=(None, None)):
        super().__init__()
        self.s_thresh = s_thresh
        self.sobel_thresh = sobel_thresh
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
            sobel1: int, sobel operator x axis lowest threshold
            sobel2: int, sobel operator x axis highest Threshold
        Returns:
            combined_binary: numpy array, binary image
        """
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=7)
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobelx = cv2.convertScaleAbs(sobelx)
        abs_sobely = cv2.convertScaleAbs(sobely)

        scaled_sobel = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[
            (scaled_sobel >= self.sobel_thresh[0])
            & (scaled_sobel <= self.sobel_thresh[1])
        ] = 1

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 1
        combined_binary = np.zeros_like(sobel_binary)
        combined_binary[(s_binary == 1) | (sobel_binary == 1)] = 1

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
        cv2.createTrackbar("sobel1", self.tuning_title, 0, 255, self.__nothing__)
        cv2.createTrackbar("sobel2", self.tuning_title, 0, 255, self.__nothing__)

        while True:
            s1 = cv2.getTrackbarPos("s1", self.tuning_title)
            s2 = cv2.getTrackbarPos("s2", self.tuning_title)
            sobel1 = cv2.getTrackbarPos("sobel1", self.tuning_title)
            sobel2 = cv2.getTrackbarPos("sobel2", self.tuning_title)
            self.s_thresh = (s1, s2)
            self.sobel_thresh = (sobel1, sobel2)
            bw = self.r2b(img)
            cv2.imshow(self.tuning_title, bw * 255)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break

        return bw

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

    def tune_imgs(self, imgs, cfg_name="config.yaml"):
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
            sx_values.append(self.sobel_thresh)

        self.s_thresh = self._get_mean_of_threshold(s_values)
        self.sobel_thresh = self._get_mean_of_threshold(sx_values)

        cfg_dict = {"s_thresh": self.s_thresh, "sobel_thresh": self.sobel_thresh}

        with open(cfg_name, "w") as cfgh:
            yaml.dump(cfg_dict, cfgh)

    def load_config(self, cfg_name):
        """
        Method to load configuration
        Arguments:
            cfg_name: str, name of the yaml configuration file
        """
        with open(cfg_name, "r") as cfgh:
            cfg = yaml.load(cfgh)
        self.s_thresh = cfg["s_thresh"]
        self.sobel_thresh = cfg["sobel_thresh"]

    def transform(self, img):

        return self.r2b(img)
