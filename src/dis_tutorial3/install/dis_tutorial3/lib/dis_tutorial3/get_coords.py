#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import math
import tf_transformations

class PoseListener(Node):
    def __init__(self):
        super().__init__('pose_listener')
        
        # Change this to 'Odometry' if using /odom
        use_amcl = True

        if use_amcl:
            self.subscription = self.create_subscription(
                PoseWithCovarianceStamped,
                '/amcl_pose',  # Topic name
                self.pose_callback,
                10
            )
            self.get_logger().info("Subscribed to /amcl_pose")
        else:
            self.subscription = self.create_subscription(
                Odometry,
                '/odom',
                self.odom_callback,
                10
            )
            self.get_logger().info("Subscribed to /odom")

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.print_pose(position.x, position.y, orientation)

    def odom_callback(self, msg: Odometry):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.print_pose(position.x, position.y, orientation)

    def print_pose(self, x, y, orientation):
        # Convert quaternion to yaw
        q = orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_deg = math.degrees(yaw)
        self.get_logger().info(f"Position -> x: {x:.2f}, y: {y:.2f} | Orientation (yaw): {yaw_deg:.2f}°")

def main(args=None):
    rclpy.init(args=args)
    node = PoseListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
