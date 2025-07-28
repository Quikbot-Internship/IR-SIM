import numpy as np
from path_planning.map_utils import lidar_to_grid
from path_planning.global_planning import GlobalPlanner
from lidar.lidar_processer import Lidar_Processer
from orca_sim.orca import ORCA_RVOPlanner
from pp_control.pure_pursuit import PurePursuit
from irsim.lib import register_behavior
from matplotlib import pyplot as plt
from coodinate_grid_utils import world_to_grid, grid_to_world

# Global constants
GRID = lidar_to_grid("map.png")
MAP_ORIGIN = (0, 0)
MAP_RES = 0.05
origin = (0, 0)
grid_width = 576
grid_height = 620

# Main behavior function for Robots
@register_behavior("diff", "pure_pursuit")
def beh_diff_pure_pursuit(ego_object, external_objects, **kwargs):
    print(f"Running behavior for {ego_object.name} with color {ego_object.color}")
    state = ego_object.state
    goal = ego_object.goal
    pos = np.array(state[:2]).flatten()

    _, max_vel = ego_object.get_vel_range()     # in .yaml file
    v_max = max_vel[0, 0]
    w_max = max_vel[1, 0]
    
    # Create Global Planner if not already created
    if not hasattr(ego_object, "planner"):
        ego_object.planner = GlobalPlanner()

    # Create ORCA avoiderance behavior if not already created
    if not hasattr(ego_object, "orca_avoidance"):
        ego_object.orca_avoidance = ORCA_RVOPlanner(
            ego_object=ego_object,
            external_objects=external_objects,
            time_horizon=3.5  # seconds to look ahead
        )

    # Create Pure Pursuit controller if not already created
    if not hasattr(ego_object, "pp_controller"):
        ego_object.pp_controller = PurePursuit(ego_object=ego_object, external_objects=external_objects)

    # If the path is not already computed, compute it
    if not hasattr(ego_object, "pp_path"):
        GRID = lidar_to_grid("map.png")

        # Convert start and goal to grid indices
        start_g = world_to_grid(pos, MAP_ORIGIN, MAP_RES, grid_height=grid_height)
        goal_pt = goal[:2].flatten()
        goal_g = world_to_grid(goal_pt, MAP_ORIGIN, MAP_RES, grid_height=grid_height)
        print(f"Start grid: {start_g}, Goal grid: {goal_g}")

        # Compute path using Dijkstra's algorithm
        path_grid = ego_object.planner.dijkstra(GRID, start_g, goal_g)
        if path_grid is None:
            return np.array([[0.0], [0.0]])
        
        # Assign path to ego_object
        ego_object.pp_path = [grid_to_world(p, MAP_ORIGIN, MAP_RES, grid_height) 
                              for p in path_grid]

        # Set the path in the Pure Pursuit controller
        ego_object.pp_controller.set_path(ego_object.pp_path)

        ### PLOT PATH FOR DEBUGGING ######
        xs, ys = zip(*ego_object.pp_path)
        plt.plot(xs, ys, ego_object.color, label='Raw')
        plt.scatter(*start_g, c='green', label='Start')
        plt.scatter(*goal_g, c='white', label='Goal')
        plt.legend()
        plt.title("Dijkstra with Path Smoothing")
        plt.draw()
        #################################################

    # Ensure robot is facing the initial heading
    # This is only done once, when the path is first computed
    if not hasattr(ego_object, "initial_heading_fixed"):
        if len(ego_object.pp_path) > 1:
            start_pos = ego_object.pp_path[0]
            next_pos = ego_object.pp_path[1]
            initial_heading = np.arctan2(next_pos[1] - start_pos[1], next_pos[0] - start_pos[0])
            ego_object.state[2, 0] = initial_heading
        ego_object.initial_heading_fixed = True

    # Create Lidar processer if not already created
    if not hasattr(ego_object, "lidar_processer"):
        ego_object.lidar_processer = Lidar_Processer(ego_object=ego_object)

    # Process Lidar points to get obstacles
    obstacles = ego_object.lidar_processer.process_lidar(ego_object.get_lidar_points())
    
   
    # Compute lookahead point and update the Pure Pursuit controller
    ind, _ = ego_object.pp_controller.search_target_index(robot_pos=pos)

    # ORCA avoidance behavior
    lookahead_point = ego_object.pp_controller.get_lookahead_point()
    goal = np.array([[lookahead_point[0]], [lookahead_point[1]], [0.0]])  # 3x1 column vector
    
    # ORCA step, pass in the goal and obstacles
    control = ego_object.orca_avoidance.compute_control(goal=goal, obstacles=obstacles) 

    # Clip to robot velocity limits
    control[0, 0] = np.clip(control[0, 0], 0.0, v_max)     # Linear velocity
    control[1, 0] = np.clip(control[1, 0], -w_max, w_max)  # Angular velocity

    # Return in original 2x1 format
    return control
