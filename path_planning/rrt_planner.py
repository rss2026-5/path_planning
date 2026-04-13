import rclpy

from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from path_planning.utils import LineTrajectory
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

import numpy as np
from scipy.ndimage import binary_dilation


class RRTNode:
    """A single node in the RRT* tree."""
    __slots__ = ['pos', 'parent', 'cost']

    def __init__(self, pos, parent=None, cost=0.0):
        self.pos = pos        # (u, v) pixel coordinates
        self.parent = parent  # RRTNode or None
        self.cost = cost      # cumulative cost from start


class RRTStarPlanner(Node):
    """RRT* sampling-based path planner. Same ROS interface as the grid-based
    PathPlan node so it can be swapped in via a different launch file."""

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value

        latched_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            latched_qos)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            latched_qos
        )

        self.pose_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.pose_cb,
            10
        )

        self.map = None
        self.current_pose = None

        # RRT* parameters (pixel space unless noted)
        self.max_iter = 5000
        self.step_size = 20            # pixels
        self.goal_bias = 0.10          # probability of sampling goal
        self.goal_tol = 15             # pixels — close enough to connect
        self.rewire_radius = 40        # pixels — neighborhood for rewiring
        self.dilation_iterations = 8

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

    def map_cb(self, msg):
        self.map = msg

    def pose_cb(self, pose):
        self.current_pose = pose.pose.pose

    def goal_cb(self, msg):
        if self.current_pose is None or self.map is None:
            if self.current_pose is None:
                self.get_logger().info('Goal callback called, but no current pose')
            if self.map is None:
                self.get_logger().info('Goal callback called, but no map')
            return
        start = (self.current_pose.position.x, self.current_pose.position.y)
        end = (msg.pose.position.x, msg.pose.position.y)
        self.plan_path(start, end, self.map)

    def plan_path(self, start_point, end_point, map_msg):
        self.get_logger().info('Path planner called')
        q = map_msg.info.origin.orientation
        self.get_logger().info(f'Map origin orientation: x={q.x}, y={q.y}, z={q.z}, w={q.w}')
        # map preprocessing (object dilation, etc.), shared with trajectory_planner.py
        res = map_msg.info.resolution
        ox = map_msg.info.origin.position.x
        oy = map_msg.info.origin.position.y
        width = map_msg.info.width
        height = map_msg.info.height
        grid = np.array(map_msg.data, dtype=np.int16).reshape((height, width))

        obstacle_mask = (grid > 50) | (grid < 0)  # occupied cells
        dilated = binary_dilation(obstacle_mask, iterations=self.dilation_iterations)
        def world_to_pixel(x, y):
            dx = x - ox
            dy = y - oy
            q = map_msg.info.origin.orientation
            yaw = np.arctan2(2.0*(q.w*q.z+q.x*q.y), 1.0-2.0*(q.y**2+q.z**2))
            cos_y = np.cos(-yaw)
            sin_y = np.sin(-yaw)
            u = int((cos_y*dx-sin_y*dy) / res)
            v = int((sin_y*dx+cos_y*dy) / res)
            # u = dx / res
            # v = dy / res
            return (u, v)

        def pixel_to_world(u, v):
            q = map_msg.info.origin.orientation
            yaw = np.arctan2(2.0*(q.w*q.z+q.x*q.y), 1.0-2.0*(q.y**2+q.z**2))
            cos_y = np.cos(yaw)
            sin_y = np.sin(yaw)
            x = (cos_y*u-sin_y*v) * res + ox
            y = (sin_y*u+cos_y*v) * res + oy
            # x = u * res + ox
            # y = v * res + oy
            return (x, y)

        def is_free(u, v):
            if u < 0 or v < 0 or u >= width or v >= height:
                return False
            return not dilated[v, u]

        def dist(m, n): #distance begween two node positions
            return ((m[0]-n[0])**2+(m[1]-n[1])**2)**0.5

        def connectible(m, n): #checks if there's a collision-free line between two node positions
            N = max(abs(n[0]-m[0]), abs(n[1]-m[1])) #number of points to check
            if N==0:
                return True
            for t in np.linspace(0, 1, N):
                u, v = round(t*m[0]+(1-t)*n[0]), round(t*m[1]+(1-t)*n[1])
                if not is_free(u, v):
                    return False
            return True

        start_px = world_to_pixel(*start_point)
        goal_px = world_to_pixel(*end_point)

        self.get_logger().info(f'Map size: {width}x{height}, resolution: {res}')
        self.get_logger().info(f'Grid dtype range: min={grid.min()}, max={grid.max()}')
        self.get_logger().info(f'Obstacle cells (grid>50): {np.sum(obstacle_mask)}, '
                                f'total cells: {width*height}')
        self.get_logger().info(f'Dilated obstacle cells: {np.sum(dilated)}')
        self.get_logger().info(f'Free cells: {np.sum(~dilated)}')
        self.get_logger().info(f'Start world: {start_point}, pixel: {start_px}')
        self.get_logger().info(f'Goal world: {end_point}, pixel: {goal_px}')
        self.get_logger().info(f'Start is_free: {is_free(*start_px)}, '
                                f'Goal is_free: {is_free(*goal_px)}')

        # RRT*
        start_node = RRTNode(pos=start_px, parent=None, cost=0.0)
        tree = [start_node]
        goal_node = None

        for iter in range(self.max_iter):
            sample = goal_px #with probability goal_bias, uniform random otherwise
            rand = np.random.random()
            if rand > self.goal_bias:
                sample = (np.random.randint(0, width-1), np.random.randint(0, height-1))
                if not is_free(sample[0], sample[1]):
                    continue

            nearest_node = start_node
            for node in tree:
                if dist(node.pos, sample) < dist(nearest_node.pos, sample):
                    nearest_node = node
            delta = (sample[0] - nearest_node.pos[0], sample[1] - nearest_node.pos[1])
            #update to next position
            next_pos = sample
            d_delta = dist(delta, [0,0])
            if d_delta > self.step_size:
                next_pos = (round(nearest_node.pos[0] + (delta[0] / d_delta) * self.step_size), round(nearest_node.pos[1] + (delta[1] / d_delta) * self.step_size))
            if not is_free(*next_pos) or not connectible(nearest_node.pos, next_pos):
                continue
            #construct neighbors (for the rewiring step)
            neighbors = [node for node in tree if dist(node.pos, next_pos) < self.rewire_radius and connectible(node.pos, next_pos)]
            #find best parent/cost among neighbors
            best_parent = nearest_node
            best_cost = nearest_node.cost + dist(nearest_node.pos, next_pos)
            for node in neighbors:
                cost = node.cost + dist(node.pos, next_pos)
                if cost < best_cost:
                    best_parent = node
                    best_cost = cost
            next_node = RRTNode(pos=next_pos, parent=best_parent, cost=best_cost)
            tree.append(next_node)

            #rewiring step
            for node in neighbors:
                new_cost = next_node.cost + dist(next_pos, node.pos)
                if new_cost < node.cost:
                    node.parent = next_node
                    node.cost = new_cost

            #check if reached goal
            if dist(next_pos, goal_px) < self.goal_tol and connectible(next_pos, goal_px):
                goal_cost = next_node.cost+dist(next_pos,goal_px)
                if goal_node is None or goal_node.cost > goal_cost:
                    goal_node = RRTNode(pos=goal_px, parent=next_node, cost=goal_cost)
                tree.append(goal_node)
                #keep running bc iterations still have room for improvement

        self.get_logger().info(f'Tree size after {self.max_iter} iterations: {len(tree)}')

        if goal_node is None:
            self.get_logger().error("RRT* failed to find a path")
            return

        #Extract path
        path = []
        node = goal_node
        while node is not None:
            path.append(node.pos)
            node = node.parent
        path.reverse()

        #Publish
        for (u, v) in path:
            self.trajectory.addPoint(pixel_to_world(u, v))

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()
        self.trajectory.clear()


def main(args=None):
    rclpy.init(args=args)
    planner = RRTStarPlanner()
    rclpy.spin(planner)
    rclpy.shutdown()
