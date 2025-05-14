#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion, Point
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from collections import deque, defaultdict
import math
import time
from sklearn.decomposition import PCA

class FaceFilter(Node):
    def __init__(self):
        super().__init__('face_filter')
        
        # ROS2 Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Marker, '/people_marker', self.marker_callback, 10)
        self.create_subscription(PointCloud2, '/oakd/rgb/preview/depth/points', self.pc_callback, 10)
        self.filtered_pub = self.create_publisher(MarkerArray, '/filtered_faces', 10)
        self.emergency_pub = self.create_publisher(Marker, '/emergency_commands', 10)

        # Configuration
        self.cluster_radius = 1.0  # meters for position clustering
        self.dest_radius = 1.0      # meters for destination similarity
        self.min_cluster_size = 5
        self.new_face_threshold = 0.5
        self.position_history = defaultdict(lambda: deque(maxlen=10))
        self.current_pc = None

        # Wall normal calculation
        self.pca = PCA(n_components=3)
        self.search_radius = 0.3  # meters around face to find wall points

    def pc_callback(self, msg):
        self.current_pc = msg

    def marker_callback(self, msg):
        if not self.current_pc:
            return

        try:
            # Transform to map frame
            map_point = self.transform_to_map(msg.pose.position, msg.header.frame_id)
            if not map_point: return

            # Calculate wall normal using PCA on nearby points
            normal = self.calculate_wall_normal(map_point.point)
            if normal is None: return

            # Calculate destination 1m perpendicular from wall
            destination = self.calculate_perpendicular_destination(map_point.point, normal)
            
            # Cluster processing
            self.process_clusters(map_point.point, destination)

        except Exception as e:
            self.get_logger().error(f"Processing error: {str(e)}")

    def calculate_wall_normal(self, face_point):
        # Extract points within radius of face position
        points = []
        for p in pc2.read_points(self.current_pc, field_names=("x", "y", "z"), skip_nans=True):
            if np.linalg.norm(np.array(p[:3]) - np.array([face_point.x, face_point.y, face_point.z])) < self.search_radius:
                points.append(p[:3])
        
        if len(points) < 10:  # Minimum points for PCA
            return None
            
        # Compute PCA and get normal
        self.pca.fit(points)
        normal = self.pca.components_[2]  # Smallest variance component
        return normal / np.linalg.norm(normal)

    def calculate_perpendicular_destination(self, face_point, normal):
        dest = Point()
        dest.x = face_point.x + normal[0] * 1.0
        dest.y = face_point.y + normal[1] * 1.0
        dest.z = 0.0  # Assume ground plane
        return dest

    def process_clusters(self, position, destination):
        # Find matching clusters
        matched_cluster = None
        for cluster_id, cluster_data in self.position_history.items():
            cluster_pos = cluster_data['position']
            cluster_dest = cluster_data['destination']
            
            pos_distance = np.linalg.norm(np.array([position.x - cluster_pos.x,
                                                   position.y - cluster_pos.y]))
            dest_distance = np.linalg.norm(np.array([destination.x - cluster_dest.x,
                                                    destination.y - cluster_dest.y]))
            
            if pos_distance < self.cluster_radius and dest_distance < self.dest_radius:
                matched_cluster = cluster_id
                break

        if matched_cluster is None:
            # New cluster emergency check
            if self.is_new_emergency(position):
                self.publish_emergency(position)
            return

        # Update cluster
        self.position_history[matched_cluster]['points'].append((position, destination))
        self.position_history[matched_cluster]['position'] = self.average_position(
            self.position_history[matched_cluster]['points'])
        self.position_history[matched_cluster]['destination'] = self.average_destination(
            self.position_history[matched_cluster]['points'])

        # Check for publication
        if len(self.position_history[matched_cluster]['points']) >= self.min_cluster_size:
            self.publish_cluster(matched_cluster)

    def is_new_emergency(self, position):
        for cluster_data in self.position_history.values():
            cluster_pos = cluster_data['position']
            distance = np.linalg.norm(np.array([position.x - cluster_pos.x,
                                              position.y - cluster_pos.y]))
            if distance < self.new_face_threshold:
                return False
        return True

    def publish_emergency(self, position):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.pose.position = position
        marker.scale.x = marker.scale.y = marker.scale.z = 0.5
        marker.color.r = 1.0
        marker.color.a = 1.0
        self.emergency_pub.publish(marker)

    def publish_cluster(self, cluster_id):
        cluster = self.position_history[cluster_id]
        if len(cluster['points']) < self.min_cluster_size:
            return

        # Create markers
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = cluster_id
        marker.type = Marker.ARROW
        marker.pose.position = cluster['position']
        marker.pose.orientation = self.get_yaw_orientation(cluster['position'], cluster['destination'])
        marker.scale.x = 0.1
        marker.scale.y = 0.2
        marker.scale.z = 0.1
        marker.color.g = 1.0
        marker.color.a = 0.7

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.filtered_pub.publish(marker_array)

    def average_position(self, points):
        avg = Point()
        avg.x = np.mean([p[0].x for p in points])
        avg.y = np.mean([p[0].y for p in points])
        return avg

    def average_destination(self, points):
        avg = Point()
        avg.x = np.mean([p[1].x for p in points])
        avg.y = np.mean([p[1].y for p in points])
        return avg

    def get_yaw_orientation(self, start, end):
        dx = end.x - start.x
        dy = end.y - start.y
        yaw = math.atan2(dy, dx)
        q = Quaternion()
        q.z = math.sin(yaw/2)
        q.w = math.cos(yaw/2)
        return q

    def transform_to_map(self, point, source_frame):
        try:
            ps = PointStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = source_frame
            ps.point = point
            return self.tf_buffer.transform(ps, "map", timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warning(f"Transform failed: {str(e)}")
            return None

def main():
    rclpy.init()
    node = FaceFilter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()