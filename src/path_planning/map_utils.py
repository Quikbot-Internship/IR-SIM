import cv2
import numpy as np
    
def lidar_to_grid(png_path, threshold=127, invert=True, morph_kernel_size=5):
    """
    Load a LiDAR PNG map and convert it into a cleaned occupancy grid.

    Args:
        png_path (str): Path to the LiDAR PNG image.
        threshold (int): Threshold to distinguish obstacles from free space.
        invert (bool): Whether to invert binary map (black = obstacle).
        morph_kernel_size (int): Size of morphological kernel to close small gaps.

    Returns:
        np.ndarray: 2D occupancy grid (1=obstacle, 0=free).
    """
    # Load image as grayscale
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {png_path}")

    # Threshold the image to binary
    _, binary = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)

    # Invert if needed (black = obstacle)
    if invert:
        occupancy_grid = 1 - binary
    else:
        occupancy_grid = binary

    # Apply morphological closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    closed = cv2.morphologyEx(occupancy_grid.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return closed

def compute_soft_costmap(occupancy_grid, robot_radius_pixels, max_extra_cost=100):
    robot_radius_meters = 0.3
    map_resolution = 0.05  # meters per pixel
    robot_radius_pixels = 50 # → 6
    free_mask = (occupancy_grid == 0).astype(np.uint8)
    dist = cv2.distanceTransform(free_mask, cv2.DIST_L2, maskSize=5)

    costmap = np.zeros_like(dist, dtype=np.float32)
    
    # Soft cost: exponential decay
    costmap = max_extra_cost * np.exp(-dist / (robot_radius_pixels / 10))
    print("Calculating costmap")
    return costmap