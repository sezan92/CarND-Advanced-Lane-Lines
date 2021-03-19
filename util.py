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
