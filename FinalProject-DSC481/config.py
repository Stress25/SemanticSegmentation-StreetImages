import cv2
import numpy as np

# ------- HSV RANGES -----------------------

hsv_ranges = {
    # ── ROAD ──────────────────────────────────────────────────────────────────
    "road": {
        "lower": np.array([0,   0,  30]),
        "upper": np.array([180, 35, 215]),
    },
    # Distant road near vanishing point (brighter from sky reflection).
    # V lower 150→135: catches dim/overcast/hazy scenes.
    "road_distant": {
        "lower": np.array([0,   0,  135]),
        "upper": np.array([180, 30, 255]),
    },

    # ── DIRT / SOIL ────────────────────────────────────────────────────────────
    "dirt": {
        "lower": np.array([2,  20,  40]),
        "upper": np.array([35, 200, 230]),
    },

    # ── VEGETATION ─────────────────────────────────────────────────────────────
    "vegetation": {
        "lower": np.array([30, 15, 15]),
        "upper": np.array([95, 255, 255]),
    },
    "vegetation_autumn": {
        "lower": np.array([10, 35, 55]),
        "upper": np.array([35, 220, 255]),
    },
    "vegetation_teal": {
        "lower": np.array([90,  20, 30]),
        "upper": np.array([130, 70, 200]),
    },

    # ── SKY ─────────────────────────────────────────────────────────────────────
    # Clear blue sky.
    # S lower 30→25: washed-out horizon sky edges.
    "sky_blue": {
        "lower": np.array([90,  25, 100]),
        "upper": np.array([130, 255, 255]),
    },
    "sky_white": {
        "lower": np.array([0,   0, 130]),
        "upper": np.array([180, 25, 255]),
    },
    # Hazy/pale-blue sky (humid or early-morning conditions).
    "sky_hazy": {
        "lower": np.array([85,  10, 150]),
        "upper": np.array([130,  60, 255]),
    },

    # ── OBSTACLE ──────────────────────────────────────────────────────────────
    "obstacle": {
        "lower": np.array([0,  0,  0]),
        "upper": np.array([180, 50, 88]),
    },
}

# ── ROI Ratios ────────────────────────────────────────────────────────────────
ROI = {
    "road_dirt":      0.15,   # skip top 10% for road/dirt (avoids sky bleed)
    "sky_end":        0.45,   # sky only in top 45%
    "obstacle_start": 0.10,
    "obstacle_end":   0.75,   # raised 0.60→0.75 to capture tall buildings
}

# ── Morphological kernel sizes ───────────────────────────────────────────────
morph = {
    "road":       {"ksize": 25, "close_iterations": 3, "open_iterations": 1},
    "dirt":       {"ksize": 15, "close_iterations": 2, "open_iterations": 2},
    "vegetation": {"ksize":  7, "close_iterations": 2, "open_iterations": 2},
    "sky":        {"ksize": 11, "close_iterations": 2, "open_iterations": 2},
    "obstacle":   {"ksize":  7, "close_iterations": 2, "open_iterations": 2},
}

# ── Overlay Colors (BGR for OpenCV, RGB for matplotlib) ──────────────────────
COLORS_BGR = {
    "road":       (128,   0, 128),
    "dirt":       (  0, 140, 255),
    "vegetation": (  0, 180,   0),
    "sky":        (255, 100,   0),
    "obstacle":   (  0,   0, 200),
}

COLORS_RGB = {
    "road":       (128,   0, 128),
    "dirt":       (255, 140,   0),
    "vegetation": (  0, 180,   0),
    "sky":        (  0, 100, 255),
    "obstacle":   (200,   0,   0),
}

# ── Blend alpha ───────────────────────────────────────────────────────────────
ALPHA = 0.45
BETA  = 0.55