import numpy as np
import math


# Parameters
k = 0.3  # look forward gain
Lfc = 4  # [m] look-ahead distance
#Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time tick
#WB = 2.9  # [m] wheel base of vehicle

class PurePursuit:
    def __init__(self, lookahead_dist=0.5, v_desired=0.45):
        #self.lookahead_dist = lookahead_dist
        self.v_desired = v_desired
        self.path = []
        self.cx = []
        self.cy = []
        self.old_nearest_point_index = None

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
        Lf = k * self.v_desired + Lfc
        Lf = max(Lf, 2.0)  # Clamp to minimum value
        
        # Search for the target point along the path
        while Lf > self.calc_distance(xc, yc, self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                #ind = len(self.cx) - 1
                break
            else:
                ind += 1

        return ind, Lf

    def compute_pure_pursuit_control(self, robot_pos, robot_theta):
        ind, Lf = self.search_target_index(robot_pos)
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
        omega = np.clip(omega, -1.5, 1.5)  # max omega (rad/s)

        # Constant or adaptive linear velocity
        v = self.v_desired

        return v, omega
'''
    def compute_pure_pursuit_control(self, robot_pos, robot_theta):
        # Get robot position
        xc = robot_pos[0]
        yc = robot_pos[1]
        #print(robot_pos, robot_theta)

        # 1. Find lookahead point (you'll need to implement this or pass it in)
        target_point, _ = self.search_target_index(robot_pos)
        print("Lookahead target:", self.path[target_point])
        target_x = self.cx[target_point]
        target_y = self.cy[target_point]

        # 2. Calculate angle to lookahead (alpha)
        x_delta = target_x - xc
        y_delta = target_y - yc
        #dot_product = x_delta * np.cos(robot_theta) + y_delta * np.sin(robot_theta)
        #print("Dot product with heading:", dot_product)
        alpha = np.arctan2(y_delta, x_delta) - robot_theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # normalize to [-pi, pi]

        # 3. Calculate distance to goal
        L_d = np.sqrt(x_delta**2 + y_delta**2)

        # 4. Calculate angular velocity (IR-SIM uses v and w)
        omega = (2 * self.v_desired * np.sin(alpha)) / L_d

        # 5. Optionally clip omega
        #max_omega = 1.5  # rad/s or whatever is reasonable
        #omega = np.clip(omega, -max_omega, max_omega)

        # 6. Return (or assign to IR-SIM agent)
        return self.v_desired, omega


        # Publish messages for ROS implementation
        #msg.drive.speed = velocity
        #msg.drive.steering_angle = steering_angle
        #pub.publish(msg)
'''

