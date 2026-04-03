import cv2
import numpy as np


def detect_edges(image_bgr, blur_ksize = 7):
    """Apply Canny edge detection to a binary mask."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return cv2.Canny(blurred, 100, 200)

def refine_mask_with_edges(mask, edges):
    """Use detected edges to refine the binary mask."""
    # clean the edges by dilating them to ensure we remove edge pixels from the mask
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    
    # Remove edge pixels from the mask
    refined_mask = cv2.bitwise_and(mask, cv2.bitwise_not(edges_dilated))
    
    return cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

def refine_all_masks(masks:dict, edges) -> dict:
   return {cls: refine_mask_with_edges(mask, edges) for cls, mask in masks.items()}

