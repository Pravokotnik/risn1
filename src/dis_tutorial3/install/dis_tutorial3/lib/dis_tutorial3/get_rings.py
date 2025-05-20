#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class RingColorSubscriber(Node):
    def __init__(self):
        super().__init__('ring_color_listener')

        # Subscribe to the ring color topic
        self.subscription = self.create_subscription(
            String,
            '/ring/color',
            self.listener_callback,
            10  # Queue size
        )
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'Received ring color: {msg.data}')


def main(args=None):
    rclpy.init(args=args)

    ring_color_subscriber = RingColorSubscriber()
    rclpy.spin(ring_color_subscriber)

    ring_color_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
