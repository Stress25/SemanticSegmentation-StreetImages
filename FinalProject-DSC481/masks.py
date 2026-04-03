import cv2
import numpy as np
from config import hsv_ranges, ROI


def hsv_mask(hsv_image, color_key: str) -> np.ndarray:
    """Create a binary mask for a specific HSV range defined in config."""
    if color_key not in hsv_ranges:
        raise ValueError(f"Color key '{color_key}' not found in hsv_ranges.")
    r = hsv_ranges[color_key]
    mask = cv2.inRange(hsv_image,
                       np.array(r["lower"], dtype=np.uint8),
                       np.array(r["upper"], dtype=np.uint8))
    if mask is None:
        raise ValueError(f"Failed to create mask for '{color_key}'.")
    return mask


def apply_roi(mask, h, row_start=0.0, row_end=1.0):
    roi = np.zeros_like(mask)
    roi[int(h * row_start):int(h * row_end), :] = 255
    return cv2.bitwise_and(mask, roi)


def road_mask(hsv_image, h):
    mask         = hsv_mask(hsv_image, "road")
    mask_distant = hsv_mask(hsv_image, "road_distant")
    combined     = cv2.bitwise_or(mask, mask_distant)
    return apply_roi(combined, h, row_start=ROI["road_dirt"])


def dirt_mask(hsv_image, h):
    mask  = hsv_mask(hsv_image, "dirt")
    count = cv2.countNonZero(mask)
    print(f"Dirt mask pixel count: {count}")
    return apply_roi(mask, h, row_start=ROI["road_dirt"])


def vegetation_mask(hsv_image, h):
    """
    Combines three sub-ranges into one vegetation mask:
    """
    mask_green  = hsv_mask(hsv_image, "vegetation")
    mask_autumn = apply_roi(hsv_mask(hsv_image, "vegetation_autumn"), h, row_end=0.70)
    mask_teal   = apply_roi(hsv_mask(hsv_image, "vegetation_teal"),   h, row_start=0.40)

    combined = cv2.bitwise_or(mask_green,  mask_autumn)
    combined = cv2.bitwise_or(combined,    mask_teal)
    return apply_roi(combined, h)


def sky_mask(hsv_image, h):
    mask_blue  = hsv_mask(hsv_image, "sky_blue")
    mask_white = hsv_mask(hsv_image, "sky_white")
    mask_hazy  = hsv_mask(hsv_image, "sky_hazy")
    combined   = cv2.bitwise_or(mask_blue,  mask_white)
    combined   = cv2.bitwise_or(combined,   mask_hazy)
    return apply_roi(combined, h, row_end=ROI["sky_end"])


def obstacle_mask(hsv_image, h):
    mask = hsv_mask(hsv_image, "obstacle")
    return apply_roi(mask, h,
                     row_start=ROI["obstacle_start"],
                     row_end=ROI["obstacle_end"])


def build_all_masks(hsv_image, h):
    return {
        "road":       road_mask(hsv_image, h),
        "dirt":       dirt_mask(hsv_image, h),
        "vegetation": vegetation_mask(hsv_image, h),
        "sky":        sky_mask(hsv_image, h),
        "obstacle":   obstacle_mask(hsv_image, h),
    }