#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped

class ClickedPointListener(Node):
    def __init__(self):
        super().__init__('clicked_point_listener')
        self.subscription = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_callback,
            10
        )
        self.get_logger().info("Subscribed to /clicked_point topic.")

    def clicked_point_callback(self, msg):
        point = msg.point
        frame = msg.header.frame_id
        
        self.get_logger().info(
            f"{point.x:.2f}, {point.y:.2f}, {point.z:.2f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = ClickedPointListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
