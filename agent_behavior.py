import numpy as np
from global_planning import dijkstra, lidar_to_grid
from pure_pursuit import PurePursuit
from irsim.lib import register_behavior
from matplotlib import pyplot as plt

# Load your occupancy grid once (only happens on import)
GRID = lidar_to_grid("map.png")
MAP_ORIGIN = (0, 0)
MAP_RES = 0.05
# Constants
#resolution = 0.05
origin = (0, 0)
grid_height = 576
grid_width = 620

# Convert world (meters) → grid index (pixels)
def world_to_grid(pt, origin, resolution):
    xg = int((pt[0] - origin[0]) / resolution)
    yg_unflipped = int((pt[1] - origin[1]) / resolution)
    yg = grid_height - 1 - yg_unflipped  # Flip y to match image coordinates
    return (xg, yg)

# Convert grid index (pixels) → world (meters)
def grid_to_world(pt, origin, resolution):
    px, py = pt
    py_flipped = grid_height - 1 - py  # Invert y index to world convention
    x_world = px * resolution + origin[0]
    y_world = py_flipped * resolution + origin[1]
    return (x_world, y_world)

@register_behavior("diff", "pure_pursuit")
def beh_diff_pure_pursuit(ego_object, external_objects, **kwargs):
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

        #Compute path using Dijkstra's algorithm
        path_grid = dijkstra(GRID, start_g, goal_g)
        if path_grid is None:
            return np.array([[0.0], [0.0]])
        
        #Assign path to ego_object
        ego_object.pp_path = [grid_to_world(p, MAP_ORIGIN, MAP_RES) 
                              for p in path_grid]

        ### Plotting the path for debugging
        xs, ys = zip(*ego_object.pp_path)
        plt.plot(xs, ys, 'green', label='Raw')
        plt.scatter(*start_g, c='green', label='Start')
        plt.scatter(*goal_g, c='white', label='Goal')
        plt.legend()
        plt.title("Dijkstra with Path Smoothing")
        plt.show()
        plt.pause(5)
        plt.close()
        ###################


        ego_object.pp_controller = PurePursuit()
        ego_object.pp_controller.set_path(ego_object.pp_path)

    # Compute Pure Pursuit control
    v, w = ego_object.pp_controller.compute_pure_pursuit_control(
        robot_pos=tuple(pos), robot_theta=heading
    )

    v = np.clip(v, 0.0, v_max)
    w = np.clip(w, -w_max, w_max)
    return np.array([[v], [w]])