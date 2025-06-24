import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq

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

def dijkstra(grid, start, goal):
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
            return reconstruct_path(parent, start, goal)

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if visited[ny, nx] or grid[ny, nx] == 1:
                continue

            step_cost = 1.4 if dx and dy else 1.0
            alt = current_dist + step_cost

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


if __name__ == "__main__":
    # Path to your LiDAR PNG map
    lidar_png_path = "map.png"

    # Create occupancy grid
    grid = lidar_to_grid(lidar_png_path)

    # Visualize the occupancy grid
    #plt.imshow(grid, cmap='gray')
    #plt.title("Occupancy Grid (1=obstacle, 0=free)")
    #plt.show()

    # Save numpy array for later use
    #np.save("occupancy_grid.npy", grid)
    #print("Occupancy grid saved to occupancy_grid.npy")

    start = (150, 400)
    goal = (400, 150)

    path = dijkstra(grid, start, goal)

    if path:
        print(f"Path found with {len(path)} points")
        import matplotlib.pyplot as plt
        plt.imshow(grid, cmap='gray')
        xs, ys = zip(*path)
        plt.plot(xs, ys, 'r-')
        plt.scatter(*start, c='green')
        plt.scatter(*goal, c='blue')
        plt.title("Dijkstra Path")
        plt.show()
    else:
        print("No path found")