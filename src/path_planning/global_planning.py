import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
from shapely.geometry import LineString
from scipy.interpolate import CubicSpline
from path_planning.map_utils import lidar_to_grid, compute_soft_costmap

class GlobalPlanner:
    def __init__(self):
        """
        Initialize the global planner.
        This class implements Dijkstra's algorithm for path planning on a 2D occupancy grid.
        """
        self.smooth_pts = 200  # Number of points for smoothing
        self.simplify_tol = 1.0  # Tolerance for path simplification

    def dijkstra(self, grid, start, goal):
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
                raw_path = self.reconstruct_path(parent, start, goal)
                simplified = self.simplify_path(raw_path, tolerance = self.simplify_tol)
                smoothed = self.smooth_path(simplified, num_points = self.smooth_pts)
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

    def reconstruct_path(self, parent, start, goal):
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

    def simplify_path(self, path, tolerance):
        """Use RDP algorithm to simplify the path."""
        if len(path) < 3:
            return path
        line = LineString(path)
        simplified = line.simplify(tolerance)
        return list(simplified.coords)

    def smooth_path(self, path, num_points):
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


## FOR TESTING PURPOSES
if __name__ == "__main__":
    grid = lidar_to_grid("map.png")

    start = (200, 300)
    goal = (420, 200)

    planner = GlobalPlanner()
    path = planner.dijkstra(grid, start, goal)

    if path:
        # Print waypoints (grid coords)
        print("Waypoints (grid coords):")
        for i, (x, y) in enumerate(path):
            print(f"{i}: ({x:.2f}, {y:.2f})")

        # Plot occupancy grid and path
        plt.imshow(grid, cmap='gray')
        xs, ys = zip(*path)
        plt.plot(xs, ys, 'green', label='Smoothed Path')
        plt.scatter(xs, ys, c='red', s=10, label='Waypoints')

        # Label waypoints
        #for i, (x, y) in enumerate(path):
        #    plt.text(x, y, str(i), color='yellow', fontsize=6)

        plt.scatter(*start, c='green', label='Start')
        plt.scatter(*goal, c='white', label='Goal')
        plt.legend()
        plt.title("Dijkstra with Path Smoothing and Waypoints")
        plt.show()
    else:
        print("No path found")