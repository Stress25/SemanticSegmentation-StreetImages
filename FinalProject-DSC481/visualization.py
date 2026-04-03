import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import COLORS_BGR, COLORS_RGB,ALPHA,BETA

# -------overlay visualization----------------
def build_overlay(image, masks:dict) -> np.ndarray:
    overlay = image.copy()
    for cls,color in COLORS_BGR.items():
       if cls in masks:
            overlay[masks[cls] == 255] = color

    return cv2.addWeighted(image, ALPHA, overlay, BETA, 0)

# ----bounding box visualization----------------
def draw_bounding_boxes(image, obstacle_mask, min_area=300):
    """Draws boxes around obstacles like cars or poles."""
    img_with_boxes = image.copy()
    contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            found_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 255), 3) # Use Yellow (BGR)
    print(f"[Debug] Drew {found_count} bounding boxes.")
    return img_with_boxes

# -----full visualization ----------------
def visualize(image, masks:dict, edges, save_path:str) -> None:

    blended = build_overlay(image, masks)

    panels = [
        ("Original + boxes", cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
        ("Road",    masks["road"], "gray"),
        ("Dirt",    masks["dirt"], "gray"),
        ("Vegetation", masks["vegetation"], "gray"),
        ("Sky",     masks["sky"], "gray"),
        ("Obstacles", masks["obstacle"], "gray"),
        ("Edges",    edges, "magma"),
        ("Overlay + boxes", cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)),
    ]

    fig,axes = plt.subplots(2,4, figsize=(16,8))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    for i, (title, img, *cmap) in enumerate(panels):
        axes[i].imshow(img, cmap=cmap[0] if cmap else None)
        axes[i].set_title(title)
        axes[i].axis('off')

#    Create the Legend in the corner of the Overlay or as a separate element
    legend_handles = [mpatches.Patch(color=np.array(color)/255, label=cls.capitalize()) 
                     for cls, color in COLORS_RGB.items()]
    fig.legend(handles=legend_handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle( 'Scene Segmentation — Road · Dirt · Vegetation · Sky · Obstacles', fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()