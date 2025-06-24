import numpy as np

class PurePursuit:
    def __init__(self, lookahead_dist=0.5, v_desired=0.3):
        """
        Args:
            lookahead_dist (float): Lookahead distance in meters.
            v_desired (float): Desired forward speed in m/s.
        """
        self.lookahead_dist = lookahead_dist
        self.v_desired = v_desired
        self.path = []
        self.last_index = 0

    def set_path(self, path):
        """
        Set the path to follow.

        Args:
            path (list of (x, y)): Path in world coordinates.
        """
        self.path = path
        self.last_index = 0


    def get_lookahead_point(path, robot_pos, lookahead_dist):
        """
        Find the first point on the path that is >= lookahead_dist from robot.

        Args:
            path (list of (x, y)): The planned global path.
            robot_pos (tuple): Robot's current position (xr, yr).
            lookahead_dist (float): Desired lookahead distance (L).

        Returns:
            (x, y) point or None if not found.
        """
        if not path:
            return None

        # Accumulate distances from current robot position
        total_dist = 0.0
        last_point = robot_pos

        for i in range(len(path)):
            pt = path[i]
            dx = pt[0] - last_point[0]
            dy = pt[1] - last_point[1]
            step_dist = np.hypot(dx, dy)
            total_dist += step_dist
            last_point = pt

            if total_dist >= lookahead_dist:
                return pt

        return path[-1]  # If no point is far enough, return last point

    def compute_pure_pursuit_control(robot_pos, robot_theta, path, lookahead_dist=0.5, v_pref=0.3):
        # Get robot position
        xc = robot_pos[0]
        yc = robot_pos[1]

        # 1. Find lookahead point (you'll need to implement this or pass it in)
        pnt = PurePursuit.get_lookahead_point(path, robot_pos, lookahead_dist)
        target_x = pnt[0]
        target_y = pnt[1]

        # 2. Calculate angle to lookahead (alpha)
        x_delta = target_x - xc
        y_delta = target_y - yc
        alpha = np.arctan2(y_delta, x_delta) - robot_theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # normalize to [-pi, pi]

        # 3. Calculate distance to goal
        L_d = np.sqrt(x_delta**2 + y_delta**2)

        # 4. Calculate angular velocity (IR-SIM uses v and w)
        omega = (2 * v_pref * np.sin(alpha)) / L_d

        # 5. Optionally clip omega
        #max_omega = 1.5  # rad/s or whatever is reasonable
        #omega = np.clip(omega, -max_omega, max_omega)

        # 6. Return (or assign to IR-SIM agent)
        return v_pref, omega


        # Publish messages for ROS implementation
        #msg.drive.speed = velocity
        #msg.drive.steering_angle = steering_angle
        #pub.publish(msg)