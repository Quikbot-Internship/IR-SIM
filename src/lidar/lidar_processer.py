from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt


class Lidar_Processer:
    def __init__(self, ego_object=None):
        self.ego_object = ego_object

    def process_lidar(self, lidar_points, max_segment_gap=0.1):
        if(lidar_points is None or len(lidar_points) < 2):
            return []

        # Transpose to (N, 2) before transforming
        pts = lidar_points.T  # (N, 2)

        # Convert to world frame (expects (N, 2))
        world_points = self.lidar_to_world(pts)

        initial_segments = self.try_split_segments(world_points, distance_threshold=0.3)

        # Merging is trivial here (one segment)
        merged_segments = self.try_merge_segments(initial_segments, max_gap=max_segment_gap)

        hulls = self.segments_to_hulls(merged_segments)
        return hulls

    def try_split_segments(self, points, distance_threshold=0.3):
        """
        Splits a list of 2D points into segments where consecutive points
        are closer than distance_threshold.
        
        Input: points shape (N, 2)
        Returns: list of np.array segments
        """
        segments = []
        current = [points[0]]

        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i-1])
            if dist > distance_threshold:
                segments.append(np.array(current))
                current = [points[i]]
            else:
                current.append(points[i])

        if current:
            segments.append(np.array(current))

        return segments

    def try_merge_segments(self, segments, max_gap):
        merged = []
        current = segments[0]
        for next_seg in segments[1:]:
            if np.linalg.norm(current[-1] - next_seg[0]) < max_gap:
                current = np.vstack((current, next_seg))
            else:
                merged.append(current)
                current = next_seg
        merged.append(current)
        return merged

    def segments_to_hulls(self, merged_segments):
        hulls = []
        for seg in merged_segments:
            if len(seg) < 3:
                continue
            try:
                hull = ConvexHull(seg)
                hull_pts = seg[hull.vertices]
                 
                hulls.append(hull_pts)
            except:
                continue
        return hulls

    def lidar_to_world(self, lidar_points):
        """
        Convert LiDAR points from robot-relative frame to world frame.
        
        Parameters:
            lidar_points: (N, 2) np.array of points in robot frame (meters)
            robot_pose: (x, y, theta) tuple
                - x, y: robot position in world (meters)
                - theta: robot orientation in radians

        Returns:
            (N, 2) np.array of points in world frame (meters)
        """
        pos = np.array(self.ego_object.state[:2]).flatten()
        x, y = pos[0], pos[1]
        theta = self.ego_object.state[2, 0]

        # 2D rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # Rotate and translate points
        world_points = (R @ lidar_points.T).T + np.array([x, y])

        return world_points
    
   