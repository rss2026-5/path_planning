import rclpy

from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from path_planning.utils import LineTrajectory
from rclpy.node import Node

import heapq
import numpy as np
from scipy.ndimage import binary_dilation

# import tf2_ros
# import tf2_geometry_msgs


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def map_cb(self, msg):
        self.map = msg

    def pose_cb(self, pose):
        self.current_pose = pose.pose.pose

    def goal_cb(self, msg):
        if self.current_pose is None or self.map is None:
            return
        
        # pose = PoseStamped()
        # pose.header.frame_id = "base_link"
        # pose.header.stamp = rclpy.time.Time().to_msg()

        # pose.pose = self.current_pose

        # # Transform into map frame
        # pose_map = self.tf_buffer.transform(pose, "map")

        start = (self.current_pose.position.x, self.current_pose.position.y)
        end   = (msg.pose.position.x, msg.pose.position.y)
        self.plan_path(start, end, self.map)

    def plan_path(self, start_point, end_point, map):
        res = map.info.resolution
        ox = map.info.origin.position.x
        oy = map.info.origin.position.y
        width = map.info.width
        height = map.info.height
        grid = np.array(map.data, dtype = np.int8).reshape((height,width))

        obstacle_mask = grid > 50  # occupied cells
        dilated = binary_dilation(obstacle_mask, iterations=8)
        def world_to_pixel(x,y):
            u = int((x - ox) / res)
            v = int((y - oy) / res)
            return (u,v)
        
        def pixel_to_world(u,v):
            x = u * res + ox
            y = v * res + oy
            return (x, y)
        
        def is_free(u, v):
            if u < 0 or v < 0 or u >= width or v >= height:
                return False
            return not dilated[v, u]

        start = world_to_pixel(*start_point)
        goal = world_to_pixel(*end_point)

        # self.get_logger().info(f"current_pose frame: {self.current_pose_frame_id}")

        # self.get_logger().info(f"Start world: {start_point}")
        # self.get_logger().info(f"Map origin: ({ox}, {oy})")
        # self.get_logger().info(f"Resolution: {res}")
        # self.get_logger().info(f"Map size: {width} x {height}")

        if not is_free(*start):
            self.get_logger().error(f"Start invalid: {start}")
            return

        if not is_free(*goal):
            self.get_logger().error(f"Goal invalid: {goal}")
            return

        prev = {start : None}
        g_score = {start : 0}
        
        # A* algorithm
        def h(a,b):
            return np.hypot(a[0] - b[0],a[1] - b[1])
        
        heap = [(h(start,goal),0,start)]
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        found = False
        while heap:
            f,g, current = heapq.heappop(heap)
            if current == goal:
                found = True
                break
            if g > g_score.get(current,float('inf')):
                continue
            for du,dv in neighbors:
                nb = (current[0] + du,current[1] + dv)
                if not is_free(*nb):
                    continue
                w = np.hypot(du,dv)
                new_g = g + w
                if new_g < g_score.get(nb,float('inf')):
                    g_score[nb] = new_g
                    prev[nb] = current
                    heapq.heappush(heap,(new_g + h(nb,goal), new_g,nb))

        if not found:
            self.get_logger().warn("No path found!")
            return

        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        for (u,v) in path:
            x,y = pixel_to_world(u,v)
            self.trajectory.addPoint(x,y)
        
        self.traj_pub.publish(self.trajectory.toPoseArray())
        
        self.trajectory.publish_viz()
        self.trajectory.clear()
        
        


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
