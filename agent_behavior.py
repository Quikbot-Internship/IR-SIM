import numpy as np
from global_planning import dijkstra, lidar_to_grid
from pure_pursuit import PurePursuit
from irsim.lib import register_behavior
import cv2
from matplotlib import pyplot as plt

# Load your occupancy grid once (only happens on import)
GRID = lidar_to_grid("map.png")

# Utility to convert world position → grid index
def world_to_grid(pt, origin, resolution, grid_height = 576):
    xg = int((pt[0] - origin[0]) / resolution)
    yg = grid_height - 1 - int((pt[1] - origin[1]) / resolution)
    return (xg, yg)


# Utility to convert grid index → world position
def grid_to_world(idx, origin, resolution, grid_height = 576):
    x_world = idx[0] * resolution + origin[0]
    y_world = (grid_height - 1 - idx[1]) * resolution + origin[1]
    return (x_world, y_world)


# Define your origin & resolution (match your YAML file)
MAP_ORIGIN = (0, 0)
MAP_RES = 0.05
# Constants
resolution = 0.05
origin = (0, 0)
grid_height = 576

@register_behavior("diff", "pure_pursuit")
def beh_diff_pure_pursuit(ego_object, external_objects, **kwargs):
    #GRID = lidar_to_grid("map.png")
    state = ego_object.state
    goal = ego_object.goal
    pos = np.array(state[:2]).flatten()
    heading = state[2, 0]

    _, max_vel = ego_object.get_vel_range()
    v_max = max_vel[0, 0]
    w_max = max_vel[1, 0]

    if not hasattr(ego_object, "pp_path"):
        GRID = lidar_to_grid("map.png")
        # Convert start and goal to grid indices
        start_g = world_to_grid(pos, MAP_ORIGIN, MAP_RES)
        goal_pt = goal[:2].flatten()
        goal_g = world_to_grid(goal_pt, MAP_ORIGIN, MAP_RES)
        print(f"Start grid: {start_g}, Goal grid: {goal_g}")


        path_grid = dijkstra(GRID, start_g, goal_g)
        if path_grid is None:
            return np.array([[0.0], [0.0]])

    # Plotting ###############
        ##plt.imshow(GRID, cmap='gray')
        xs, ys = zip(*path_grid)
#        plt.plot(xs, ys, 'green', label='Raw')

 #       plt.scatter(*start_g, c='green', label='Start')
  #      plt.scatter(*goal_g, c='black', label='Goal')
   #     plt.legend()
     #   plt.title("Dijkstra with Path Smoothing")
    #    plt.show()
        #############
        
        # Convert grid path back to world points
        ego_object.pp_path = [grid_to_world(p, MAP_ORIGIN, MAP_RES) 
                              for p in path_grid]
        

        ego_object.pp_controller = PurePursuit()
        ego_object.pp_controller.set_path(ego_object.pp_path)
        #for point in ego_object.pp_path:
        #    print(f"({point[0]:.2f}, {point[1]:.2f})")

    # Compute control
    v, w = ego_object.pp_controller.compute_pure_pursuit_control(
        robot_pos=tuple(pos), robot_theta=heading
    )
    #print(v, w)

    

    v = np.clip(v, 0.0, v_max)
    w = np.clip(w, -w_max, w_max)

    return np.array([[v], [w]])
