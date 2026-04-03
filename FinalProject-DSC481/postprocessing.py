import cv2
import numpy as np
from config import morph


def clean_mask(mask, ksize, close_iterations: int = 3, open_iterations: int = 2):
    """Apply morphological close then open to fill holes and remove noise."""
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel, iterations=open_iterations)
    return cleaned


def clean_all_masks(masks: dict) -> dict:
    return {cls: clean_mask(mask, **morph[cls]) for cls, mask in masks.items()}


# ── Overlap resolution ────────────────────────────────────────────────────────
def resolve_overlaps(masks: dict) -> dict:
    """Resolves overlaps between classes based on spatially-aware priority rules:
    """
    for cls, mask in masks.items():
        if mask is None:
            raise ValueError(f"Mask for class '{cls}' is None.")

    m = {k: v.copy() for k, v in masks.items()}
    h = list(m.values())[0].shape[0]

    # Spatial split masks
    mid        = int(h * 0.55)
    upper_zone = np.zeros(m["road"].shape, dtype=np.uint8)
    upper_zone[:mid, :] = 255
    lower_zone = cv2.bitwise_not(upper_zone)

    # 1. Sky wins over everything
    not_sky = cv2.bitwise_not(m["sky"])
    for key in ("road", "vegetation", "dirt", "obstacle"):
        m[key] = cv2.bitwise_and(m[key], not_sky)

    # 2. Spatially-aware road / vegetation priority
    #    upper image: vegetation wins → strip road pixels that overlap with veg
    #    lower image: road wins       → strip veg pixels that overlap with road
    veg_in_upper  = cv2.bitwise_and(m["vegetation"], upper_zone)
    road_in_lower = cv2.bitwise_and(m["road"],        lower_zone)
    m["road"]       = cv2.bitwise_and(m["road"],       cv2.bitwise_not(veg_in_upper))
    m["vegetation"] = cv2.bitwise_and(m["vegetation"], cv2.bitwise_not(road_in_lower))

    not_road = cv2.bitwise_not(m["road"])
    not_veg  = cv2.bitwise_not(m["vegetation"])

    # 3. Road beats dirt everywhere
    m["dirt"] = cv2.bitwise_and(m["dirt"], not_road)

    # 4. Vegetation beats dirt
    m["dirt"] = cv2.bitwise_and(m["dirt"], not_veg)

    # 5. Obstacle zone logic (rows 10–60 %)
    obs_zone = np.zeros_like(m["road"])
    obs_zone[int(h * 0.10):int(h * 0.60), :] = 255

    # Keep obstacles only in their zone
    m["obstacle"] = cv2.bitwise_and(m["obstacle"], obs_zone)
    # Obstacles lose to vegetation
    m["obstacle"] = cv2.bitwise_and(m["obstacle"], not_veg)
    # Within the zone, obstacles beat road and dirt
    obs_active = m["obstacle"].copy()
    not_obs    = cv2.bitwise_not(obs_active)
    m["road"]  = cv2.bitwise_and(m["road"], not_obs)
    m["dirt"]  = cv2.bitwise_and(m["dirt"], not_obs)

    return m


# ── Road refinement ───────────────────────────────────────────────────────────
def refine_road_with_contours(road_mask, top_k: int = 3):
    """
    Fills the top-K largest road contours; applies convex hull only when ratio<1.5.
    This prevents overfilling non-road areas while still capturing the main road blob.
    """
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 8))
    morph_road = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph_road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    refined = np.zeros_like(road_mask)
    if not contours:
        return refined

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:top_k]:
        cv2.drawContours(refined, [cnt], -1, 255, thickness=cv2.FILLED)
        cnt_area  = cv2.contourArea(cnt)
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if cnt_area > 0 and hull_area / cnt_area < 1.5:
            cv2.drawContours(refined, [hull], -1, 255, thickness=cv2.FILLED)

    return refined


# ── Largest connected component ───────────────────────────────────────────────
def largest_connected_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8) * 255


# ── Full post-processing pipeline ─────────────────────────────────────────────
def postprocess_masks(masks: dict) -> dict:
    """
    Step 1 — resolve_overlaps 

    Step 2 — refine_road_with_contours

    Step 3 — obstacle / road separation
        Obstacles do not bleed into refined road.

    Step 4 — clean_all_masks  (morphological open + close)
    
    Step 5 — resolve_overlaps )
      
    Step 6 — largest_connected_component for sky
    """
    masks = resolve_overlaps(masks)                          # Step 1
    masks["road"] = refine_road_with_contours(masks["road"], top_k=3)  # Step 2
    not_obs = cv2.bitwise_not(masks["obstacle"])
    masks["road"] = cv2.bitwise_and(masks["road"], not_obs)  # Step 3
    masks = clean_all_masks(masks)                           # Step 4
    masks = resolve_overlaps(masks)                          # Step 5
    masks["sky"] = largest_connected_component(masks["sky"]) # Step 6
    return masks