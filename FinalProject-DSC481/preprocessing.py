import cv2
import numpy as np
from config import hsv_ranges, ROI, morph

def Load_Image(image_path):
    # load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    return image

def Apply_CLAHE(image):
    """
    Adaptive Histogram Equalization to balance shadows and highlights.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# apply gaussian blur to reduce noise
def Apply_Gaussian_Blur(image, ksize=5):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def Convert_To_HSV(image):
    # convert to HSV color space
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def preprocess_image(image_path):
    # Full preprocessing pipeline

    # image   : original BGR image
    # hsv     : HSV image (blurred)
    # h, w    : image dimensions

    img = Load_Image(image_path)
    image = Apply_CLAHE(img)
    blurred = Apply_Gaussian_Blur(image)
    hsv_image = Convert_To_HSV(blurred)
    h, w = image.shape[:2]

    return image, hsv_image, h, w