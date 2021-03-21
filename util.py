import os

import cv2


def imread(image_name):
    """
    Reads image and returns RGB numpy array
    Parameters:
        image_name: Image path , str
    Returns:
        img: numpy array
    """
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
