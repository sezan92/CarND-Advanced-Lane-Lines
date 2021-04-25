import cv2
import numpy as np


def get_histogram(binary):
    return np.sum(binary[binary.shape[0] // 2 :, :], axis=0)


def get_binary_3d(binary):
    return np.dstack((binary, binary, binary))


def get_midpoint(histogram):
    return np.int(histogram.shape[0] // 2)


def get_leftx_base(histogram, midpoint):
    return np.argmax(histogram[:midpoint])


def get_rightx_base(histogram, midpoint):
    return np.argmax(histogram[midpoint:]) + midpoint


def get_window_height(binary, nwindows):
    return np.int(binary.shape[0] // nwindows)


def get_indices(binary, nwindows, margin, minpix, nonzerox, nonzeroy):
    window_height = get_window_height(binary, nwindows)
    histogram = get_histogram(binary)
    midpoint = get_midpoint(histogram)
    leftx_base = get_leftx_base(histogram, midpoint)
    rightx_base = get_rightx_base(histogram, midpoint)
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary.shape[0] - (window + 1) * window_height
        win_y_high = binary.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

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

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    return left_lane_inds, right_lane_inds


def detect_lane(binary, nwindows, margin, minpix):

    out_img = get_binary_3d(binary)
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds, right_lane_inds = get_indices(
        binary, nwindows, margin, minpix, nonzerox, nonzeroy
    )

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary, nwindows=20, margin=60, minpix=25):
    leftx, lefty, rightx, righty, out_img = detect_lane(
        binary, nwindows, margin, minpix
    )

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    ploty = np.linspace(0, binary.shape[0] - 1, binary.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit


def draw_lane(img, binary, left_fitx, right_fitx, ploty, pt):
    #   # At first lets create an empty mask
    mask = np.zeros_like(binary).astype(np.uint8)
    mask_color = np.dstack((mask, mask, mask))

    # Lets make an array of left lane points and right lane points. Here we will have to transpose after stacking them , as in the opencv images have y dimension or number of rows are counted first.
    points_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    points_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    points = np.hstack((points_left, points_right))
    # Lets fill the polygon of the points with color green ,and the color the lanes as Red and Blue respectively
    mask_color = cv2.fillPoly(mask_color, np.int32([points]), (0, 255, 0))
    mask_color = cv2.polylines(
        mask_color,
        np.int32([points_left]),
        isClosed=False,
        color=(255, 0, 0),
        thickness=15,
    )
    mask_color = cv2.polylines(
        mask_color,
        np.int32([points_right]),
        isClosed=False,
        color=(0, 0, 255),
        thickness=15,
    )
    # Lets transform the mask into original perspective
    mask_color_previous = pt.inverse_transform(mask_color)
    img_lane = cv2.addWeighted(img, 1, mask_color_previous, 0.5, 0)
    return img_lane
