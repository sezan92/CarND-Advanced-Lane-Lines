import argparse

from camera import Camera
from preprocess import BinaryTransformer, PerspectiveTransformer
from util import load_img_from_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        help="directory to the images to set the binary threshold",
        required=True,
    )
    parser.add_argument(
        "--cfg_filename",
        type=str,
        help="configureation file name. Default: binary_config.yaml",
        default="binary_config.yaml",
    )
    parser.add_argument(
        "--camera_cfg_filename",
        type=str,
        help="camera configuration filename. Default: camera_config.yaml",
        default="camera_config.yaml",
    )
    parser.add_argument(
        "--pt_cfg_filename",
        type=str,
        help="Perspective Transformation configuration file. Default: pt_config.yaml",
        default="pt_config.yaml",
    )

    args = parser.parse_args()
    img_dir = args.image_dir
    cfg_filename = args.cfg_filename
    camera_cfg_filename = args.camera_cfg_filename
    pt_cfg_filename = args.pt_cfg_filename

    camera = Camera()
    pt = PerspectiveTransformer()
    bt = BinaryTransformer()
    camera.load_config(camera_cfg_filename)
    pt.load_config(pt_cfg_filename)
    imgs = [pt.transform(camera.transform(img)) for img in load_img_from_dir(img_dir)]
    bt.tune_imgs(imgs, cfg_filename)
