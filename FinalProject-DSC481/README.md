Street Scene Segmentation Pipeline

Course: CSC 381/481 - Intro to Image Processing 

Name: Kritika Bhat
DePaul ID: 2178208
Term: Winter 2025-26 

Project Description
This prototype implements an image processing pipeline to segment street-level images into five distinct semantic classes: Road, Dirt, Vegetation, Sky, and Obstacles (cars, poles, buildings). It utilizes traditional computer vision techniques rather than deep learning to achieve high-speed segmentation through color-space analysis and morphological refinement.

Features & Image Processing Techniques
Color Space Conversion: Transformation from BGR to HSV for robust color segmentation.

Adaptive Equalization: Uses CLAHE to balance lighting and shadows in diverse environments.

Multi-Class Masking: Custom HSV ranges for specific environmental conditions (e.g., blue vs. hazy sky, autumn vs. green foliage).

Spatial Constraints: ROI (Region of Interest) filtering to reduce false positives (e.g., sky isn't on the ground).

Edge Refinement: Integration of Canny Edge Detection to sharpen segment boundaries.

Morphological Operations: Extensive use of Closing and Opening to remove noise and fill gaps.

Object Tracking: Bounding box generation for detected obstacles.

Installation & Dependencies
This project requires Python 3.12 and the following libraries:

1. opencv-python
2. numpy
3. matplotlib

How to Run
To run the segmentation on a single image:

--> python main.py --image path/to/your/image.jpg

To save the result and disable edge refinement:

--> python main.py --image path/to/image.jpg --save output_result.png --no-edges

File Structure
1. main.py: Entry point for the pipeline.
2. config.py: Contains hyper-parameters (HSV ranges, ROI ratios, Morphological kernels).
3. masks.py: Logic for generating class-specific binary masks.
4. preprocessing.py: Image loading, CLAHE, and blurring.
5. postprocessing.py: Overlap resolution and morphological cleaning.
6. edges.py: Canny edge detection and mask refinement logic.
7. visualization.py: Generates the 8-panel comparison figure and overlays.