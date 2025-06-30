import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
from shapely.geometry import LineString
from scipy.interpolate import CubicSpline

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

    # Zero out areas far enough from obstacles
    #costmap[dist > robot_radius_pixels] = 0

    # Visualization (optional)
    #plt.imshow(costmap, cmap='hot')
    #plt.colorbar(label='Penalty')
    #plt.title("Exponential Gradient Costmap")
    #plt.show()
    print("Calculating costmap")
    return costmap

def dijkstra(grid, start, goal, simplify_tol=1.0, smooth_pts=400):
    """
    Run Dijkstra's algorithm on a 2D occupancy grid.

    Args:
        grid: 2D numpy array where 1 = obstacle, 0 = free
        start: (x, y) tuple in grid coordinates
        goal: (x, y) tuple in grid coordinates

    Returns:
        path: list of (x, y) grid points from start to goal, or None if no path
    """
    height, width = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    dist = np.full_like(grid, np.inf, dtype=float)
    parent = np.full((height, width, 2), -1, dtype=int)
    costmap = compute_soft_costmap(grid, robot_radius_pixels=4)

    dist[start[1], start[0]] = 0
    heap = [(0, start)]

    # 8-connected grid
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1),
             (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while heap:
        current_dist, (x, y) = heapq.heappop(heap)

        if visited[y, x]:
            continue
        visited[y, x] = True

        if (x, y) == goal:
            raw_path = reconstruct_path(parent, start, goal)
            simplified = simplify_path(raw_path, tolerance=simplify_tol)
            smoothed = smooth_path(simplified, num_points=smooth_pts)
            return smoothed

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if visited[ny, nx] or grid[ny, nx] == 1:
                continue

            step_cost = 1.4 if dx and dy else 1.0
            penalty = costmap[ny, nx]
            alt = current_dist + step_cost + penalty

            if alt < dist[ny, nx]:
                dist[ny, nx] = alt
                parent[ny, nx] = [x, y]
                heapq.heappush(heap, (alt, (nx, ny)))

    return None  # No path found

def reconstruct_path(parent, start, goal):
    path = []
    x, y = goal
    while (x, y) != start:
        path.append((x, y))
        x, y = parent[y, x]
        if x == -1:  # Invalid parent
            return None
    path.append(start)
    path.reverse()
    return path

def simplify_path(path, tolerance=2.0):
    """Use RDP algorithm to simplify the path."""
    if len(path) < 3:
        return path
    line = LineString(path)
    simplified = line.simplify(tolerance)
    return list(simplified.coords)

def smooth_path(path, num_points=200):
    """Fit cubic spline to path."""
    if len(path) < 2:
        return path

    path = np.array(path)
    distances = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    distances = np.insert(distances, 0, 0)
    t = np.linspace(0, distances[-1], num_points)

    cs_x = CubicSpline(distances, path[:, 0])
    cs_y = CubicSpline(distances, path[:, 1])

    x_smooth = cs_x(t)
    y_smooth = cs_y(t)

    return list(zip(x_smooth, y_smooth))

if __name__ == "__main__":
    grid = lidar_to_grid("map.png")

    start = (200, 300)
    goal = (420, 200)

    path = dijkstra(grid, start, goal)
    print(path)
    if path:
        print(f"Raw path length: {len(path)}")

        # Step 1: Simplify
        simplified_path = simplify_path(path, tolerance=1.0)
        print(f"Simplified path length: {len(simplified_path)}")

        # Step 2: Smooth
        smooth = smooth_path(simplified_path, num_points=400)
        print(f"Smoothed path length: {len(smooth)}")

        # Plotting
        plt.imshow(grid, cmap='gray')
        xs, ys = zip(*path)
        plt.plot(xs, ys, 'green', label='Raw')

        xs, ys = zip(*simplified_path)
        #plt.plot(xs, ys, 'blue', label='Simplified')

        xs, ys = zip(*smooth)
        #plt.plot(xs, ys, 'red', label='Smoothed')

        plt.scatter(*start, c='green', label='Start')
        plt.scatter(*goal, c='black', label='Goal')
        plt.legend()
        plt.title("Dijkstra with Path Smoothing")
        plt.show()
    else:
        print("No path found")
        