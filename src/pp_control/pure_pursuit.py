import numpy as np
import math


class PurePursuit:
    def __init__(self, ego_object, external_objects):
        """ Initialize the Pure Pursuit controller."""
        self.path = []
        self.cx = []
        self.cy = []
        self.old_nearest_point_index = None
        self.current_index = 0  # Current index in the path
        self.k = 0.30           # look forward gain
        self.Lfc = .50          # [m] look-ahead distance
        self.min_Lf = 0.5       # minimum look-ahead distance 
        self.max_Lf = 1.25      # maxiumum look-ahead distance
        #Kp = 1.0               # speed proportional gain
        #self.dt = 0.1          # [s] time tick
        #WB = 2.9               # [m] wheel base of vehicle

        _, max_vel = ego_object.get_vel_range()     # in .yaml file
        self.v_desired = max_vel[0, 0]
        self.w_max = max_vel[1, 0]
        self.external_objects = external_objects

    def set_path(self, path):
        """
        Set the path and precompute cx, cy.
        Args:
            path (list of (x, y)): Path in world coordinates.
        """
        self.path = path
        self.cx = [p[0] for p in path]      #x coordinates of the path
        self.cy = [p[1] for p in path]      #y coordinates of the path
        self.old_nearest_point_index = None

    def calc_distance(self, robot_x, robot_y, point_x, point_y):
        dx = robot_x - point_x
        dy = robot_y - point_y
        return math.hypot(dx, dy)
    
    def search_target_index(self, robot_pos):
        xc, yc = robot_pos

        #Find index of the nearest point on the path
        if self.old_nearest_point_index is None:
            dx = [xc - icx for icx in self.cx]  
            dy = [yc - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = self.calc_distance(xc, yc, self.cx[ind], self.cy[ind])

            while True:
                if (ind + 1) >= len(self.cx):
                    break
                distance_next_index = self.calc_distance(xc, yc, self.cx[ind + 1], self.cy[ind + 1])

                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        # Update lookahead distance
        Lf = self.k * self.v_desired + self.Lfc
        Lf = max(Lf, self.min_Lf, self.max_Lf)  # Clamp to minimum value
        
        # Search for the target point along the path
        while (Lf > self.calc_distance(xc, yc, self.cx[ind], self.cy[ind])) or not self.validate_lookahead_point(ind):
            if (ind + 1) >= len(self.cx):
                break
            else:
                ind += 1
                
        self.current_index = ind
        return ind, Lf

    def compute_pure_pursuit_control(self, robot_pos, robot_theta):
        ind, Lf = self.search_target_index(robot_pos)
        self.current_index = ind
        # Get robot position
        xc = robot_pos[0]
        yc = robot_pos[1]


        if ind < len(self.cx):
            tx = self.cx[ind]
            ty = self.cy[ind]
        else:
            tx, ty = self.cx[-1], self.cy[-1]
            ind = len(self.cx) - 1

        # Compute angle to target
        dx = tx - xc
        dy = ty - yc
        angle_to_target = math.atan2(dy, dx)
        heading_error = angle_to_target - robot_theta

        # Normalize to [-pi, pi]
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        # Compute angular velocity
        K_ang = 2.0
        omega = K_ang * heading_error
        omega = np.clip(omega, -self.w_max, self.w_max)  # max omega (rad/s)

        # Constant or adaptive linear velocity
        v = self.v_desired

        return v, omega

    def validate_lookahead_point(self, idx):
        """Ensure the lookahead point is not occupied."""
        for obj in self.external_objects:
            if obj.name.startswith('robot'):
                radius = obj.radius
                obj_x, obj_y = obj.state[0], obj.state[1]
                lookahead_x, lookahead_y = self.cx[idx], self.cy[idx]
                distance = np.hypot(lookahead_x - obj_x, lookahead_y - obj_y)

                if distance < radius:
                    return False
        return True
    
    def get_lookahead_point(self):
        return self.cx[self.current_index], self.cy[self.current_index]