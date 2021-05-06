import argparse

import cv2

from camera import Camera
from preprocess import PerspectiveTransformer
from util import imread

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        help="image path to get the perstpective transform configuration",
        required=True,
    )
    parser.add_argument(
        "--cfg_filename",
        type=str,
        help="configureation file name. Default: pt_config.yaml",
        default="pt_config.yaml",
    )
    parser.add_argument(
        "--camera_cfg_filename",
        type=str,
        help="camera configuration filename. Default: camera_config.yaml",
        default="camera_config.yaml",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        help="threshold for perspective transform . Depents on the image shape. Default:400",
        default=400,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="to plot or not to plot. Default: False",
        default=False,
    )
    args = parser.parse_args()
    image_name = args.image
    img = imread(image_name)
    IMG_SIZE = img.shape[:2]
    camera = Camera(IMG_SIZE)
    camera.load_config(args.camera_cfg_filename)
    img = camera.transform(img)
    pt = PerspectiveTransformer(IMG_SIZE)
    pt.tune(img, cfg_name=args.cfg_filename)
    if args.plot:
        img_transformed = pt.transform(img)
        img_transformed = cv2.cvtColor(img_transformed, cv2.COLOR_RGB2BGR)
        cv2.imshow("transformed image", img_transformed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
