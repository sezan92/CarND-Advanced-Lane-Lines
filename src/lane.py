from collections import deque

import numpy as np


class LaneDetector:
    def __init__(self, nwindows=20, margin=60, minpix=25):
        """
        LaneDetector Class
            arguments:
                nwindows: number of windows
                margin: width of a window
                minpix: minimum numper of pixels in a line
        """
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix

    def set_nwindows(self, nwindows):
        self.nwindows = nwindows

    def set_margin(self, margin):
        self.margin = margin

    def set_minpix(self, minpix):
        self.minpix = minpix

    def get_histogram(self, binary):
        return np.sum(binary[binary.shape[0] // 2 :, :], axis=0)

    def get_binary_3d(self, binary):
        return np.dstack((binary, binary, binary))

    def get_midpoint(self, histogram):
        return np.int(histogram.shape[0] // 2)

    def get_leftx_base(self, histogram, midpoint):
        return np.argmax(histogram[:midpoint])

    def get_rightx_base(self, histogram, midpoint):
        return np.argmax(histogram[midpoint:]) + midpoint

    def get_window_height(self, binary):
        return np.int(binary.shape[0] // self.nwindows)

    def get_indices(self, binary, nonzerox, nonzeroy):
        window_height = self.get_window_height(binary)
        histogram = self.get_histogram(binary)
        midpoint = self.get_midpoint(histogram)
        leftx_base = self.get_leftx_base(histogram, midpoint)
        rightx_base = self.get_rightx_base(histogram, midpoint)
        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            win_y_low = binary.shape[0] - (window + 1) * window_height
            win_y_high = binary.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            good_left_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        return left_lane_inds, right_lane_inds

    def detect_lane(self, binary):

        out_img = self.get_binary_3d(binary)
        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds, right_lane_inds = self.get_indices(binary, nonzerox, nonzeroy)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def fit(self, binary):
        leftx, lefty, rightx, righty, out_img = self.detect_lane(binary)
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except TypeError:
            left_fit = self.previous_left_fit
            right_fit = self.previous_right_fit
            import cv2

            cv2.imshow("problem binary frame", binary * 255)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            print(rightx, righty)

        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        ploty = np.linspace(0, binary.shape[0] - 1, binary.shape[0])
        try:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty
        self.previous_left_fit = left_fit
        self.previous_right_fit = right_fit
        self.previous_leftx = leftx
        self.previous_lefty = lefty
        self.previous_rightx = rightx
        self.previous_righty = righty
        return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit


class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype="float")
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        # recent number of values
        self.n = 15
        # polynomial coefficients for the most recent fit
        self.current_fit = deque(maxlen=self.n)

    def fit(self, fit):
        if self.best_fit is not None:
            self.diffs = fit - self.best_fit
            if (
                abs(self.diffs[0]) > 0.01
                or abs(self.diffs[1]) > 1.0
                or abs(self.diffs[2]) > 100
            ):
                self.detected = False
            else:
                self.current_fit.append(fit)
                self.best_fit = np.average(self.current_fit, axis=0)
                self.detected = True
        else:
            self.best_fit = fit
            self.current_fit.append(fit)
            self.detected = True

    def get_fitx(self, ploty):
        best_fitx = (
            self.best_fit[0] * ploty ** 2 + self.best_fit[1] * ploty + self.best_fit[2]
        )
        return best_fitx
