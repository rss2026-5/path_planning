#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class SafetyController(Node):
    """
    Emergency-only safety watchdog for path following.

    Publishes a hard stop to the VESC mux safety topic when an obstacle is
    critically close. Stays silent otherwise so the mux timeout restores
    navigation control automatically.

    Two detection zones:
      - Forward  (|angle| < FORWARD_CONE):  stops when closer than EMERGENCY_DIST
      - Side     (FORWARD_CONE < |angle| < SIDE_CONE): stops when closer than
                 SIDE_DIST, preventing the car body from clipping corners.

    Hysteresis prevents chattering: the emergency is only cleared once all
    obstacle distances exceed their respective thresholds + RELEASE_MARGIN.
    """

    # Forward zone — head-on collision prevention
    FORWARD_CONE = np.pi / 3        # ±60 deg
    EMERGENCY_DIST = 0.25           # hard-stop threshold [m]
    RELEASE_MARGIN = 0.15           # hysteresis margin [m]

    # Side zone — wheel / corner clipping prevention
    SIDE_CONE = 5 * np.pi / 12      # ±75 deg outer boundary
    SIDE_DIST = 0.10                # side hard-stop threshold [m]

    # Use a robust percentile (not min) to ignore single noisy returns
    RANGE_PERCENTILE = 10

    def __init__(self):
        super().__init__("safety_controller")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("safety_topic", "/vesc/low_level/input/safety")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        safety_topic = self.get_parameter("safety_topic").get_parameter_value().string_value

        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.safety_pub = self.create_publisher(AckermannDriveStamped, safety_topic, 10)

        self.in_emergency = False
        self.front_dist = float("inf")
        self.side_dist = float("inf")

        self.get_logger().info(
            f"Safety controller ready — scan: {scan_topic}, safety: {safety_topic}"
        )

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        valid = np.isfinite(ranges) & (ranges > 0.0)

        # Forward zone: ±60°
        fwd_mask = valid & (np.abs(angles) < self.FORWARD_CONE)
        self.front_dist = (
            float(np.percentile(ranges[fwd_mask], self.RANGE_PERCENTILE))
            if fwd_mask.any()
            else float("inf")
        )

        # Side zone: 60° – 75° on both sides
        side_mask = valid & (np.abs(angles) >= self.FORWARD_CONE) & (
            np.abs(angles) < self.SIDE_CONE
        )
        self.side_dist = (
            float(np.percentile(ranges[side_mask], self.RANGE_PERCENTILE))
            if side_mask.any()
            else float("inf")
        )

        # Trigger emergency
        front_danger = self.front_dist < self.EMERGENCY_DIST
        side_danger = self.side_dist < self.SIDE_DIST

        if not self.in_emergency and (front_danger or side_danger):
            self.in_emergency = True
            if front_danger:
                self.get_logger().warn(
                    f"EMERGENCY STOP (front): obstacle {self.front_dist:.2f} m"
                )
            else:
                self.get_logger().warn(
                    f"EMERGENCY STOP (side): obstacle {self.side_dist:.2f} m"
                )

        # Clear emergency only when both zones are safe (hysteresis)
        elif self.in_emergency:
            front_clear = self.front_dist > (self.EMERGENCY_DIST + self.RELEASE_MARGIN)
            side_clear = self.side_dist > (self.SIDE_DIST + self.RELEASE_MARGIN)
            if front_clear and side_clear:
                self.in_emergency = False
                self.get_logger().info(
                    f"Emergency cleared — front: {self.front_dist:.2f} m, "
                    f"side: {self.side_dist:.2f} m"
                )

        if self.in_emergency:
            stop = AckermannDriveStamped()
            stop.header.stamp = self.get_clock().now().to_msg()
            stop.header.frame_id = "base_link"
            stop.drive.speed = 0.0
            stop.drive.steering_angle = 0.0
            self.safety_pub.publish(stop)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
