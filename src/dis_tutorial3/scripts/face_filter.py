#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, Point, PoseStamped
import numpy as np
from collections import deque
import time
from rclpy.qos import QoSProfile, QoSHistoryPolicy

class FaceFilterNode(Node):
    def __init__(self):
        super().__init__('face_filter')
        
        # TF2 setup with larger buffer
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=30))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Time synchronization setup
        self.set_parameters([rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, False)])
        
        # Marker publishers
        qos = QoSProfile(
            depth=5,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        self.marker_publisher = self.create_publisher(MarkerArray, '/waypoints', qos)
        self.filtered_pub = self.create_publisher(Marker, '/filtered_people_marker', 10)
        
        # Face tracking
        self.face_history = deque(maxlen=20)
        self.distance_threshold = 0.3
        self.time_window = 2.0
        self.safe_distance = 0.5  # meters to stay back from face
        
        # Transform readiness
        self.transform_ready = False
        self.create_timer(0.1, self.check_transform_ready)
        
        self.get_logger().info("Face filter node initialized!")

    def get_synchronized_time(self):
        """Get time that works with current time source"""
        if self.get_parameter('use_sim_time').value:
            return self.get_clock().now()
        return rclpy.time.Time(seconds=0)  # Use latest available for real clock

    def check_transform_ready(self):
        if not self.transform_ready:
            try:
                transform = self.tf_buffer.lookup_transform(
                    "map",
                    "base_link",
                    self.get_synchronized_time(),
                    timeout=rclpy.duration.Duration(seconds=0.5))
                
                self.get_logger().info("Transform available! Creating subscriber...")
                self.transform_ready = True
                self.marker_sub = self.create_subscription(
                    Marker, '/people_marker', self.marker_callback, 10)
                self.get_logger().info("Subscribed to /people_marker topic.")
                    
            except TransformException:
                self.get_logger().info("Waiting for transform...", throttle_duration_sec=5)

    def marker_callback(self, msg):
        try:
            # Get current robot position
            robot_pose = self.get_current_robot_pose()
            if not robot_pose:
                return

            # Transform face detection to map frame
            face_point = self.transform_to_map(msg.pose.position, "base_link")
            if not face_point:
                return

            # Calculate safe approach point
            safe_point = self.calculate_safe_point(robot_pose.pose.position, face_point.point)
            
            # Create visualization markers
            if self.is_new_face(face_point.point):
                filtered_marker = Marker()
                filtered_marker.header.frame_id = "map"
                filtered_marker.header.stamp = self.get_clock().now().to_msg()
                filtered_marker.id = len(self.face_history)
                filtered_marker.pose.position = safe_point
                
                marker_array = self.create_visualization_markers(
                    robot_pose.pose.position,
                    face_point.point,
                    safe_point
                )
                
                
                
                # Publish markers
                self.marker_publisher.publish(marker_array)
                self.filtered_pub.publish(filtered_marker)
                self.face_history.append((face_point.point, time.time()))
                self.get_logger().info(
                    f"Safe point calculated: {safe_point.x:.2f}, {safe_point.y:.2f}"
                )

        except Exception as e:
            self.get_logger().error(f'Error in callback: {str(e)}')

    def get_current_robot_pose(self):
        """Get current robot position in map frame"""
        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                self.get_synchronized_time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            
            pose = PoseStamped()
            pose.header = transform.header
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            return pose
            
        except TransformException as ex:
            self.get_logger().warning(f'Robot pose temporarily unavailable: {ex}',
                                    throttle_duration_sec=1.0)
            return None

    def transform_to_map(self, point, source_frame):
        """Transform a point to map frame with proper time handling"""
        try:
            point_stamped = PointStamped()
            point_stamped.header.stamp = self.get_synchronized_time().to_msg()
            point_stamped.header.frame_id = source_frame
            point_stamped.point = point
            
            return self.tf_buffer.transform(
                point_stamped,
                "map",
                timeout=rclpy.duration.Duration(seconds=0.1))
                
        except TransformException as ex:
            self.get_logger().warning(f'Transform temporarily unavailable: {ex}',
                                    throttle_duration_sec=1.0)
            return None

    def calculate_safe_point(self, robot_pos, face_pos):
        """Calculate point 0.5m back from face toward robot"""
        # Convert to numpy arrays for vector math
        robot = np.array([robot_pos.x, robot_pos.y, robot_pos.z])
        face = np.array([face_pos.x, face_pos.y, face_pos.z])
        
        # Calculate direction vector from face to robot
        direction = robot - face
        distance = np.linalg.norm(direction)
        
        # Normalize and scale to safe distance
        if distance > 0:
            direction = direction / distance
            scaled_direction = direction * min(self.safe_distance, distance)
        else:
            scaled_direction = np.zeros(3)
        
        # Calculate safe point
        safe_point = Point()
        safe_point.x = face_pos.x + scaled_direction[0]
        safe_point.y = face_pos.y + scaled_direction[1]
        safe_point.z = face_pos.z + scaled_direction[2]
        
        return safe_point

    def create_visualization_markers(self, robot_pos, face_pos, safe_point):
        """Create markers for visualization"""
        marker_array = MarkerArray()
        
        # 1. Face position (red sphere)
        face_marker = Marker()
        face_marker.header.frame_id = "map"
        face_marker.header.stamp = self.get_clock().now().to_msg()
        face_marker.ns = "face_detection"
        face_marker.id = 0
        face_marker.type = Marker.SPHERE
        face_marker.action = Marker.ADD
        face_marker.pose.position = face_pos
        face_marker.scale.x = 0.3
        face_marker.scale.y = 0.3
        face_marker.scale.z = 0.3
        face_marker.color.r = 1.0
        face_marker.color.a = 1.0
        marker_array.markers.append(face_marker)
        
        # 2. Safe approach point (green sphere)
        safe_marker = Marker()
        safe_marker.header = face_marker.header
        safe_marker.ns = "safe_point"
        safe_marker.id = 1
        safe_marker.type = Marker.SPHERE
        safe_marker.action = Marker.ADD
        safe_marker.pose.position = safe_point
        safe_marker.scale.x = 0.3
        safe_marker.scale.y = 0.3
        safe_marker.scale.z = 0.3
        safe_marker.color.g = 1.0
        safe_marker.color.a = 1.0
        marker_array.markers.append(safe_marker)
        
        # 3. Line from robot to face (blue line)
        line_marker = Marker()
        line_marker.header = face_marker.header
        line_marker.ns = "path_line"
        line_marker.id = 2
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.points = [robot_pos, safe_point, face_pos]
        line_marker.scale.x = 0.05  # Line width
        line_marker.color.b = 1.0
        line_marker.color.a = 0.5
        marker_array.markers.append(line_marker)
        
        return marker_array

    def is_new_face(self, point):
        """Check if this is a new face detection"""
        current_pos = np.array([point.x, point.y, point.z])
        current_time = time.time()
        
        for (stored_pos, timestamp) in self.face_history:
            stored_array = np.array([stored_pos.x, stored_pos.y, stored_pos.z])
            # self.get_logger().info(f"Checking {stored_array} against {current_pos}. Difference is {np.linalg.norm(current_pos - stored_array)}")
            if np.linalg.norm(current_pos - stored_array) < self.distance_threshold:
                return False
        return True

def main():
    rclpy.init()
    node = FaceFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()