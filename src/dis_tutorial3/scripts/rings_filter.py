#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
import numpy as np
from collections import deque, defaultdict, Counter
import time
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy

class RingsFilterNode(Node):
    def __init__(self):
        super().__init__('rings_filter')
        
        # TF2 setup with larger buffer
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Ring tracking
        self.distance_threshold = 1.0  # meters
        self.time_window = 2.0  # seconds
        self.robot_position_threshold = 1.0 
        
        # Ring dictionary for tracking
        # Structure: {ring_id: {'position': (x, y, z), 'colors': Counter(), 'last_updated': timestamp}}
        self.rings_dict = {}
        self.next_ring_id = 0
        
        # Publisher
        self.filtered_pub = self.create_publisher(
            Marker,
            '/filtered_rings_marker',
            10)
            
        # Subscriber (created AFTER transform is available)
        self.transform_ready = False
        self.create_timer(0.1, self.check_transform_ready)  # Check every 100ms

        qos = QoSProfile(
            depth=5,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        self.marker_publisher = self.create_publisher(MarkerArray, '/waypoints', qos)
        
        self.get_logger().info("Rings filter node initialized!")

    def get_synchronized_time(self):
        if self.get_parameter('use_sim_time').value:
            return self.get_clock().now()
        return rclpy.time.Time(seconds=0)  #

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

    def get_color_name(self, msg):
        r = msg.color.r
        g = msg.color.g
        b = msg.color.b

        self.get_logger().info(f"Color: {r}, {g}, {b}")

        if (r == 1.0 and g == 0.0 and b == 0.0):
            return "red"
        elif (r == 0.0 and g == 1.0 and b == 0.0):
            return "green"
        elif (r == 0.0 and g == 0.0 and b == 1.0):
            return "blue"
        elif (r == 0.0 and g == 0.0 and b == 0.0):
            return "black"
        else:
            return "unknown"

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
            
            # Check if nan or inf
            if (point_map.point.x == float('nan') or point_map.point.y == float('nan') or point_map.point.z == float('nan') or
                point_map.point.x == float('inf') or point_map.point.y == float('inf') or point_map.point.z == float('inf')):
                return
            
            # Log
            self.get_logger().info(f"Transformed point: {point_map.point.x}, {point_map.point.y}, {point_map.point.z}")

            # Get color name
            color_name = self.get_color_name(msg)
            if color_name == "unknown":
                self.get_logger().info("Unknown color, skipping...")
                return

            self.get_logger().info(f"Color: {color_name}")
            
            # Process the ring
            self.process_ring_detection(point_map, color_name)
                
        except TransformException as ex:
            self.get_logger().error(f'Transform failed: {ex}', throttle_duration_sec=1.0)

    def process_ring_detection(self, point_map, color_name):
        current_pos = np.array([point_map.point.x, point_map.point.y, point_map.point.z])
        current_time = time.time()
        
        # Get current robot position in map frame
        try:
            robot_transform = self.tf_buffer.lookup_transform(
                "map", 
                "base_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            
            robot_position = np.array([
                robot_transform.transform.translation.x,
                robot_transform.transform.translation.y,
                robot_transform.transform.translation.z
            ])
        except TransformException:
            self.get_logger().error("Failed to get robot position")
            return
        
        # Check if this ring is already in our dictionary
        matching_ring_id = None
        for ring_id, ring_data in self.rings_dict.items():
            ring_pos = ring_data['position']
            stored_array = np.array([ring_pos[0], ring_pos[1], ring_pos[2]])
            
            # Distance between current detection and stored ring
            ring_distance = np.linalg.norm(current_pos - stored_array)
            
            # Check if we have a robot position stored for this ring
            if 'robot_position' in ring_data:
                stored_robot_pos = ring_data['robot_position']
                robot_pos_distance = np.linalg.norm(robot_position - stored_robot_pos)
                
                # If both ring is close and robot is in similar position, it's likely the same ring
                if ring_distance < self.distance_threshold or (
                        ring_distance < self.distance_threshold * 1.5 and  # Slightly relaxed distance
                        robot_pos_distance < self.robot_position_threshold):
                    matching_ring_id = ring_id
                    break
            else:
                # Fall back to just distance check if no robot position stored
                if ring_distance < self.distance_threshold:
                    matching_ring_id = ring_id
                    break
        
        if matching_ring_id is not None:
            # Update existing ring data
            self.rings_dict[matching_ring_id]['colors'][color_name] += 1
            self.rings_dict[matching_ring_id]['last_updated'] = current_time
            # Update robot position
            self.rings_dict[matching_ring_id]['robot_position'] = robot_position
            
            # Get the most common color
            most_common_color = self.rings_dict[matching_ring_id]['colors'].most_common(1)[0][0]
            count = self.rings_dict[matching_ring_id]['colors'][most_common_color]
            total = sum(self.rings_dict[matching_ring_id]['colors'].values())
            
            self.get_logger().info(f"Updated ring {matching_ring_id}: most common color is {most_common_color} "
                                f"({count}/{total} detections)")
            
            # Update the ring position (averaging could be implemented here)
            # For now, we just keep the latest position
            self.rings_dict[matching_ring_id]['position'] = (point_map.point.x, point_map.point.y, point_map.point.z)
            
            # Publish the updated ring
            self.publish_filtered_ring(point_map, most_common_color, matching_ring_id)
            
        else:
            # Create new ring entry
            new_ring_id = self.next_ring_id
            self.next_ring_id += 1
            
            self.rings_dict[new_ring_id] = {
                'position': (point_map.point.x, point_map.point.y, point_map.point.z),
                'colors': Counter({color_name: 1}),
                'last_updated': current_time,
                'robot_position': robot_position  # Store robot position for new ring
            }
            
            self.get_logger().info(f"New ring detected! ID: {new_ring_id}, Color: {color_name}")
            
            # Publish the new ring - FIX: Use new_ring_id instead of ring_id
            self.publish_filtered_ring(point_map, color_name, new_ring_id)

    def publish_filtered_ring(self, point_map, color_name, ring_id):
        marker_array = MarkerArray()

        filtered_marker = Marker()
        filtered_marker.header.frame_id = "map"
        filtered_marker.header.stamp = self.get_clock().now().to_msg()
        filtered_marker.type = Marker.SPHERE
        filtered_marker.id = ring_id
        filtered_marker.scale.x = 0.3
        filtered_marker.scale.y = 0.3
        filtered_marker.scale.z = 0.3
        
        # Set color based on color_name
        filtered_marker.color.a = 1.0
        if color_name == "red":
            filtered_marker.color.r = 1.0
            filtered_marker.color.g = 0.0
            filtered_marker.color.b = 0.0
        elif color_name == "green":
            filtered_marker.color.r = 0.0
            filtered_marker.color.g = 1.0
            filtered_marker.color.b = 0.0
        elif color_name == "blue":
            filtered_marker.color.r = 0.0
            filtered_marker.color.g = 0.0
            filtered_marker.color.b = 1.0
        elif color_name == "black":
            filtered_marker.color.r = 0.0
            filtered_marker.color.g = 0.0
            filtered_marker.color.b = 0.0
        
        filtered_marker.pose.position = point_map.point
        
        self.filtered_pub.publish(filtered_marker)

        marker_array.markers.append(filtered_marker)

        self.get_logger().info("najbi narisal markr na sliko")
        #self.marker_publisher.publish(marker_array)

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