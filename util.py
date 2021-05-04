import os

import cv2
import numpy as np


def imread(image_name, colorspace="RGB"):
    """
    Reads image and returns RGB numpy array
    Parameters:
        image_name: Image path , str
        colorspace: RGB or BGR, str
    Returns:
        img: numpy array
    """
    img = cv2.imread(image_name)
    if colorspace == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif colorspace == "BGR":
        pass
    else:
        raise "{} colorspcace not identified".format(colorspace)

    return img


def load_img_from_dir(dir_name):
    """
    str -> list
    Reads images from directory and returns the image arrays list

    Parameters:
        dir_name: str, directory name
    Returns:
        imgs: list, list of image arrays
    """

    img_paths = [
        os.path.join(dir_name, img_path)
        for img_path in os.listdir(dir_name)
        if img_path.endswith("jpg")
    ]

    return [imread(img_path) for img_path in img_paths]


def draw_lane(img, binary, left_fitx, right_fitx, ploty, pt):
    """
    draws lane on a given image baesed on binary image
    Paraeters:
        left_fitx: list, x indices of left line
        right_fitx: list, x indices of right line
        ploty: y indices for plottting
        pt: PerspectiveTransformer, transforms the drawn lanes on the currect perspective
    Returns:
        img_lane: np.ndarray, image with lane drawn
    """
    #   # At first lets create an empty mask
    mask = np.zeros_like(binary).astype(np.uint8)
    mask_color = np.dstack((mask, mask, mask))

    # Lets make an array of left lane points and right lane points.
    # Here we will have to transpose after stacking them , as in the opencv images have y dimension or
    # number of rows are counted first.
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


def measure_curvature_pos(ploty, left_fitx, right_fitx, binary):
    """
    Measures curvature position and radius based on given parameters
    Parameters:
        ploty: list, y indices for fitting
        left_fitx: list, x indices for fitting for left line
        right_fitx: list, x indices for fitting for right line
    Return:
        left_curverad: float, left curve radius
        right_curverad: float, right curve radius
        vehicle_position: float, vehicle position compared to the left curve
    """
    xm_per_pix = 3.7 / 550
    ym_per_pix = 3.1 / 720

    left_fit = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    y_eval = np.max(ploty)

    left_curverad = (
        (1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5
    ) / np.absolute(2 * left_fit[0])
    right_curverad = (
        (1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5
    ) / np.absolute(2 * right_fit[0])

    H, W = binary.shape
    H = H * ym_per_pix
    left_lane_intercept = left_fit[0] * H ** 2 + left_fit[1] * H + left_fit[2]
    right_lane_intercept = right_fit[0] * H ** 2 + right_fit[1] * H + right_fit[2]

    lane_mid_point = (left_lane_intercept + right_lane_intercept) / 2

    vehicle_mid_point = W / 2 * xm_per_pix
    vehicle_position = lane_mid_point - vehicle_mid_point

    return left_curverad, right_curverad, vehicle_position
