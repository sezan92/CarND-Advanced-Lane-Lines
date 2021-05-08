import argparse
import os

import cv2
import matplotlib.pyplot as plt

from camera import Camera
from lane import LaneDetector, Line
from preprocess import BinaryTransformer, PerspectiveTransformer
from util import (draw_lane, load_img_from_dir, measure_curvature_pos,
                  preprocess_frame, prior_search)


def detect_lane_video(video_name, ld, transformers):
    """
    Detects lane in a video
    parameters:
        video_name: str, path of the video
        ld: LaneDetector, Lanedetector object. Detects lane from a binary image
        transformers: list, list of transformers to transform the image before working
        on it
    returns:
        None
    """
    output_video_name = video_name.split(".")[0] + "_output.avi"
    cap = cv2.VideoCapture(video_name)

    # Lets take the width and height of the video to create the ```VideoWriter``` object for output of the video
    # I took help for the following code from this
    # [link](https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(
        output_video_name,
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        10,
        (frame_width, frame_height),
    )

    # Let's get the coefficients for the first frame

    ret, frame = cap.read()
    binary_frame = preprocess_frame(frame, transformers)
    # Line Class for tracking

    # As suggested in the lessons, I am declaring the line class to track the lanes

    left_line = Line()
    right_line = Line()
    ret = True
    while ret:
        if not left_line.detected or not right_line.detected:
            frame_output, left_fitx, right_fitx, ploty, left_fit, right_fit = ld.fit(
                binary_frame
            )
        else:
            left_fitx, right_fitx, ploty, left_fit, right_fit = prior_search(
                binary_frame, left_fit, right_fit, margin=150
            )
        left_line.fit(left_fit)
        right_line.fit(right_fit)
        left_best_fitx = left_line.get_fitx(ploty)
        right_best_fitx = right_line.get_fitx(ploty)
        if abs(abs(left_fitx[-1] - right_fitx[-1])) < 100:
            left_line.detected = False
            right_line.detected = False

        frame_lane = draw_lane(
            frame, binary_frame, left_best_fitx, right_best_fitx, ploty, pt
        )
        left_curverad, right_curverad, vehicle_position = measure_curvature_pos(
            ploty, left_best_fitx, right_best_fitx, binary_frame
        )
        curv_radius = (left_curverad + right_curverad) / 2
        text = "Curve radius {:04.2f} m".format(curv_radius)
        cv2.putText(
            frame_lane,
            text,
            (50, 70),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        text = "vehicle position w.r.t center {:04.2f} m".format(vehicle_position)
        cv2.putText(
            frame_lane,
            text,
            (50, 100),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        out.write(frame_lane)
        ret, frame = cap.read()
        if ret:
            binary_frame = preprocess_frame(frame, transformers)

    cap.release()
    out.release()


def detect_lane_dir(img_dir, ld, transformers):
    imgs = load_img_from_dir(img_dir)
    output_dir = os.path.join(img_dir, "lane_detected")
    output_dir = os.makedirs(output_dir, exist_ok=True)
    camera = transformers[0]
    pt = transformers[1]
    bt = transformers[2]
    for i, img in enumerate(imgs):
        img = camera.transform(img)
        img_pt_transformed = pt.transform(img)
        binary = bt.r2b(img_pt_transformed)
        out_img, left_fitx, right_fitx, ploty, _, _ = ld.fit(binary)
        img_lane = draw_lane(img, binary, left_fitx, right_fitx, ploty, pt)
        left_curverad, right_curverad, vehicle_position = measure_curvature_pos(
            ploty, left_fitx, right_fitx, binary
        )
        curv_radius = (left_curverad + right_curverad) / 2

        text = "Curve radius {:04.2f} m".format(curv_radius)
        cv2.putText(
            img_lane,
            text,
            (50, 70),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        text = "vehicle position w.r.t center {:04.2f} m".format(vehicle_position)
        cv2.putText(
            img_lane,
            text,
            (50, 100),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title("Original Image")
        ax2.imshow(img_pt_transformed)
        ax2.set_title("Transformed Image")
        ax3.imshow(out_img, cmap="gray")
        ax3.set_title("Binary")
        ax4.imshow(img_lane, cmap="gray")
        ax4.set_title("Detected Lane")
        plt.savefig(os.path.join(output_dir, f"{i}.jpg"))


def is_video(filename):
    """
    Checks if the filename is video or not
    arguments:
        filename: str, filename
    returns:
        bool, True or False
    """
    video_formats = ["mp4", "avi"]
    for video_format in video_formats:
        return filename.endswith(video_format)
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="Input video or directory of images to detect lanes.",
        required=True,
    )
    parser.add_argument(
        "--camera_cfg_filename",
        type=str,
        help="camera configuration file. Default: camera_config.yaml",
        default="camera_config.yaml",
    )
    parser.add_argument(
        "--pt_cfg_filename",
        type=str,
        default="pt_config.yaml",
        help="Perspective Transformation configuration file. Default: pt_config.yaml",
    )
    parser.add_argument(
        "--binary_cfg_filename",
        type=str,
        default="binary_config.yaml",
        help="Binary Transformation configuration file. Default: bt_config.yaml",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[720, 1280],
        help="image shape. Default: (720, 1280)",
    )
    args = parser.parse_args()
    input_ = args.input
    img_shape = tuple(args.img_size)
    camera_cfg_filename = args.camera_cfg_filename
    pt_cfg_filename = args.pt_cfg_filename
    bt_cfg_filename = args.binary_cfg_filename

    camera = Camera()
    camera.load_config(camera_cfg_filename)
    pt = PerspectiveTransformer(img_shape)
    pt.load_config(pt_cfg_filename)
    bt = BinaryTransformer()
    bt.load_config(bt_cfg_filename)
    ld = LaneDetector()
    transformers = [camera, pt, bt]
    if is_video(input_):
        detect_lane_video(input_, ld, transformers)
    elif os.path.isdir(input_):
        detect_lane_dir(input_, ld, transformers)
