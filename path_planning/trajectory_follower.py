import numpy as np
import rclpy

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from visualization_msgs.msg import Marker
from .utils import LineTrajectory

class PurePursuit(Node):
    """Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed."""

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        # Tunable parameters
        self.lookahead = 1.5          # lookahead distance [m]
        self.speed = 1.0              # forward speed [m/s]
        self.wheelbase_length = 0.325 # RACECAR wheelbase [m]
        self.max_steer = 0.34         # max steering angle [rad] (~19.5 deg)

        # Stop when this close to the final waypoint
        self.goal_tolerance = 0.5  # [m]

        self.initialized_traj = False
        self.trajectory = LineTrajectory(self, "/followed_trajectory")

        # Numpy arrays built from trajectory for fast vectorized math
        self.traj_points = None   # (N, 2) array of waypoints
        self.seg_starts = None    # (N-1, 2) segment start points
        self.seg_ends = None      # (N-1, 2) segment end points
        self.seg_vecs = None      # (N-1, 2) segment vectors (end - start)
        self.seg_lens_sq = None   # (N-1,) squared segment lengths
        self.cum_dist = None      # (N,) cumulative arc-length at each waypoint

        self.pose_sub = self.create_subscription(
            Odometry, self.odom_topic, self.pose_callback, 1)
        self.traj_sub = self.create_subscription(
            PoseArray, "/trajectory/current", self.trajectory_callback, 1)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 1)

        # Debug: visualize the current lookahead target in RViz
        self.lookahead_pub = self.create_publisher(
            Marker, "/pure_pursuit/lookahead", 1)

    # ------------------------------------------------------------------ #
    #  Trajectory reception
    # ------------------------------------------------------------------ #
    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        pts = np.array(self.trajectory.points)  # (N, 2)
        if len(pts) < 2:
            self.get_logger().warn("Trajectory has fewer than 2 points, ignoring.")
            self.initialized_traj = False
            return

        self.traj_points = pts
        self.seg_starts = pts[:-1]
        self.seg_ends = pts[1:]
        self.seg_vecs = self.seg_ends - self.seg_starts
        self.seg_lens_sq = np.sum(self.seg_vecs ** 2, axis=1)  # (N-1,)

        # Cumulative arc length at each waypoint
        seg_lens = np.sqrt(self.seg_lens_sq)
        self.cum_dist = np.zeros(len(pts))
        self.cum_dist[1:] = np.cumsum(seg_lens)

        self.initialized_traj = True

    # ------------------------------------------------------------------ #
    #  Main control loop (called on every odometry message)
    # ------------------------------------------------------------------ #
    def pose_callback(self, odometry_msg):
        if not self.initialized_traj:
            return

        # Extract car pose in map frame
        pos = odometry_msg.pose.pose.position
        orient = odometry_msg.pose.pose.orientation
        car_x, car_y = pos.x, pos.y
        car_pos = np.array([car_x, car_y])

        # Yaw from quaternion
        yaw = self._quat_to_yaw(orient)

        # Check if we've reached the goal
        goal = self.traj_points[-1]
        if np.linalg.norm(car_pos - goal) < self.goal_tolerance:
            self._publish_drive(0.0, 0.0)
            return

        # 1) Find the nearest point on the trajectory (vectorized)
        nearest_seg_idx, nearest_t = self._find_nearest_segment(car_pos)

        # 2) Find the lookahead point by searching forward from the nearest segment
        lookahead_pt = self._find_lookahead_point(car_pos, nearest_seg_idx, nearest_t)

        if lookahead_pt is None:
            # No lookahead intersection found — drive toward the last waypoint
            lookahead_pt = self.traj_points[-1]

        # Publish lookahead marker for RViz debugging
        self._publish_lookahead_marker(lookahead_pt)

        # 3) Transform lookahead point into car's local frame
        dx = lookahead_pt[0] - car_x
        dy = lookahead_pt[1] - car_y
        local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy

        # If lookahead point is behind the car, just drive straight (will correct)
        if local_x <= 0.0:
            self._publish_drive(self.speed, 0.0)
            return

        # 4) Pure pursuit steering law
        # curvature = 2 * local_y / L_d^2
        L_d_sq = local_x ** 2 + local_y ** 2
        curvature = 2.0 * local_y / L_d_sq
        steering_angle = np.arctan(self.wheelbase_length * curvature)

        # Clamp steering
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        self._publish_drive(self.speed, steering_angle)

    # ------------------------------------------------------------------ #
    #  Find the nearest segment and parameter t ∈ [0,1] on that segment
    # ------------------------------------------------------------------ #
    def _find_nearest_segment(self, car_pos):
        """Vectorized: project car_pos onto every segment, return (seg_idx, t)."""
        # Vector from each segment start to the car
        v = car_pos - self.seg_starts  # (N-1, 2)

        # Parameter t of the projection, clamped to [0, 1]
        # t = dot(v, seg_vec) / |seg_vec|^2
        dots = np.sum(v * self.seg_vecs, axis=1)  # (N-1,)
        # Avoid division by zero for degenerate segments
        safe_lens_sq = np.maximum(self.seg_lens_sq, 1e-12)
        t = np.clip(dots / safe_lens_sq, 0.0, 1.0)  # (N-1,)

        # Closest point on each segment
        proj = self.seg_starts + t[:, np.newaxis] * self.seg_vecs  # (N-1, 2)

        # Distance from car to each projected point
        dists = np.linalg.norm(car_pos - proj, axis=1)  # (N-1,)

        seg_idx = int(np.argmin(dists))
        return seg_idx, t[seg_idx]

    # ------------------------------------------------------------------ #
    #  Find the lookahead point by circle-segment intersection
    # ------------------------------------------------------------------ #
    def _find_lookahead_point(self, car_pos, start_seg, start_t):
        """Walk forward along the trajectory from the nearest segment and find
        the first intersection between the lookahead circle and a segment that
        is ahead of the nearest point."""
        n_segs = len(self.seg_starts)
        r = self.lookahead

        for i in range(start_seg, n_segs):
            p1 = self.seg_starts[i]
            p2 = self.seg_ends[i]

            pt = self._circle_line_intersection(car_pos, r, p1, p2)
            if pt is not None:
                # If this is the starting segment, only accept intersections
                # that are ahead of (or at) the nearest point's t
                if i == start_seg:
                    # Compute t of the intersection on this segment
                    seg_vec = p2 - p1
                    len_sq = self.seg_lens_sq[i]
                    if len_sq < 1e-12:
                        continue
                    t_int = np.dot(pt - p1, seg_vec) / len_sq
                    if t_int < start_t - 0.01:
                        # Intersection is behind the nearest point; skip
                        continue
                return pt

        return None

    def _circle_line_intersection(self, center, radius, p1, p2):
        """Find the furthest-along-segment intersection of a circle (center, radius)
        with the line segment p1->p2.  Returns the intersection point (np array)
        closest to p2 (i.e. furthest ahead), or None."""
        d = p2 - p1
        f = p1 - center

        a = np.dot(d, d)
        b = 2.0 * np.dot(f, d)
        c = np.dot(f, f) - radius * radius

        discriminant = b * b - 4.0 * a * c

        if discriminant < 0 or a < 1e-12:
            return None

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        # We want the furthest valid intersection (largest t in [0, 1])
        best_t = None
        for t in [t2, t1]:
            if 0.0 <= t <= 1.0:
                best_t = t
                break  # t2 >= t1, so first valid is the largest

        if best_t is None:
            return None

        return p1 + best_t * d

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _quat_to_yaw(q):
        """Extract yaw from a geometry_msgs Quaternion."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _publish_drive(self, speed, steering_angle):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

    def _publish_lookahead_marker(self, point):
        """Publish a small sphere at the lookahead target for RViz."""
        if self.lookahead_pub.get_subscription_count() == 0:
            return
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "/map"
        marker.ns = "pure_pursuit"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = 0.2
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
        self.lookahead_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
