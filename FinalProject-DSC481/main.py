"""
main.py Scene Segmentation Pipeline
======================================
Segments an input street image into five classes:
  Road | Dirt | Vegetation | Sky | Obstacles (buildings,cars, etc.)

Usage
-----
    python main.py --image path/to/image.png
    python main.py --image path/to/image.png --save output.png
"""

import argparse
from preprocessing  import preprocess_image
from masks          import build_all_masks
from postprocessing import postprocess_masks
from visualization  import visualize, build_overlay,draw_bounding_boxes
from edges         import detect_edges, refine_all_masks


def run_segmentation(image_path: str, save_path: str = None,use_edges: bool = False) -> dict:
    """
    End-to-end segmentation pipeline.

    """
    # 1. Load & preprocess
    print("[main] Loading and preprocessing image")
    image, hsv, h, w = preprocess_image(image_path)

    # 2. Generate raw masks
    print("[main] Building segmentation masks ")
    masks = build_all_masks(hsv, h)

    # 3. Post-process (overlap resolution, morphology, largest component)
    print("[main] Post-processing masks")
    masks = postprocess_masks(masks)

    edges = None
    if use_edges:
        print("[main] Detecting edges for refinement")
        edges = detect_edges(image)
        masks = refine_all_masks(masks, edges)

    # 4. Visualize
    print("[main] Rendering results")
    boxed_image = draw_bounding_boxes(image, masks["obstacle"],min_area=500)
    visualize(boxed_image, masks, edges , save_path=save_path)
    # 5. Build overlay image for downstream use
    overlay = build_overlay(image, masks)

    print("[main] Done.")
    return {"masks": masks, "edges": edges, "overlay": overlay}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Street scene segmentation")
    parser.add_argument("--image",    required=True, help="Path to input image")
    parser.add_argument("--save",     default=None,  help="Path to save output figure")
    parser.add_argument("--no-edges", action="store_true",
                        help="Disable Canny edge refinement")
    args = parser.parse_args()

    run_segmentation(
        image_path=args.image,
        save_path=args.save,
        use_edges=not args.no_edges,
    )

