#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
import numpy as np
from collections import deque
import time

class RingsFilterNode(Node):
    def __init__(self):
        super().__init__('rings_filter')
        
        # TF2 setup with larger buffer
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Face tracking
        self.face_history = deque(maxlen=20)
        self.distance_threshold = 0.5  # meters
        self.time_window = 2.0  # seconds
        
        # Publisher
        self.filtered_pub = self.create_publisher(
            Marker,
            '/filtered_rings_marker',
            10)
            
        # Subscriber (created AFTER transform is available)
        self.transform_ready = False
        self.create_timer(0.1, self.check_transform_ready)  # Check every 100ms
        
        self.get_logger().info("Rings filter node initialized!")

    def check_transform_ready(self):
        if not self.transform_ready:
            try:
                # Check if transform is available
                self.tf_buffer.lookup_transform(
                    "map",
                    "base_link",
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1))
                
                self.get_logger().info("Transform available! Creating subscriber...")
                self.transform_ready = True
                self.marker_sub = self.create_subscription(
                    Marker,
                    '/ring_marker',
                    self.marker_callback,
                    10)
                self.get_logger().info("Subscriber to '/ring_marker' created.")
                    
            except TransformException:
                self.get_logger().info("Waiting for transform from 'base_link' to 'map'...", 
                                     throttle_duration_sec=5)

    def marker_callback(self, msg):
        try:
            # Get the latest available transform time
            latest_time = self.tf_buffer.get_latest_common_time(
                "base_link",
                "map"
            )
            
            # Create PointStamped with synchronized time
            point_camera = PointStamped()
            point_camera.header.stamp = latest_time.to_msg()
            point_camera.header.frame_id = "base_link"
            point_camera.point = msg.pose.position
            
            # Transform using synchronized time
            point_map = self.tf_buffer.transform(
                point_camera,
                "map",
                timeout=rclpy.duration.Duration(seconds=0.1))
            
            # Log
            self.get_logger().info(f"Transformed point: {point_map.point.x}, {point_map.point.y}, {point_map.point.z}")
            
            # Filter and publish
            if self.is_new_face(point_map):
                filtered_marker = Marker()
                filtered_marker.header.frame_id = "map"
                filtered_marker.header.stamp = self.get_clock().now().to_msg()
                filtered_marker.type = Marker.SPHERE
                filtered_marker.id = len(self.face_history)
                filtered_marker.scale.x = 0.3
                filtered_marker.scale.y = 0.3
                filtered_marker.scale.z = 0.3
                filtered_marker.color.r = 1.0
                filtered_marker.color.a = 1.0
                filtered_marker.pose.position = point_map.point
                
                self.filtered_pub.publish(filtered_marker)
                self.face_history.append((point_map.point, time.time()))

                self.get_logger().info(f"Tocka: {point_map.point}")
                
        except TransformException as ex:
            self.get_logger().error(f'Transform failed: {ex}', throttle_duration_sec=1.0)

    def is_new_face(self, point_map):
        current_pos = np.array([point_map.point.x, point_map.point.y, point_map.point.z])
        current_time = time.time()
        
        for (stored_pos, timestamp) in self.face_history:
            if current_time - timestamp > self.time_window:
                continue
                
            stored_array = np.array([stored_pos.x, stored_pos.y, stored_pos.z])
            if np.linalg.norm(current_pos - stored_array) < self.distance_threshold:
                return False
                
        return True

def main():
    rclpy.init()
    node = RingsFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()