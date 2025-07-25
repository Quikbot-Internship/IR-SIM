from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt


class Lidar_Processer:
    def __init__(self, ego_object=None):
        self.ego_object = ego_object

    def process_lidar(self, points, max_segment_gap=0.3):
        if(points is None or len(points) < 2):
            return []

        pts = points.T  # (N, 2)

        # Treat all points as one single segment
        segments = [pts]

        # Merging is trivial here (one segment)
        merged_segments = self.try_merge_segments(segments, max_gap=max_segment_gap)

        hulls = self.segments_to_hulls(merged_segments)

        return hulls

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

    