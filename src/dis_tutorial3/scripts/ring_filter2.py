#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, PoseStamped, Point, PoseWithCovarianceStamped, Vector3Stamped, Vector3
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import ColorRGBA
import numpy as np
from collections import deque, defaultdict
import math
import time
from dis_tutorial3.msg import Task, RingMsg
from tf2_geometry_msgs import do_transform_point, do_transform_vector3

from message_filters import Subscriber
from std_msgs.msg import Header  # Add this import at the top of your file

from tf2_msgs.msg import TFMessage
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from concurrent.futures import ThreadPoolExecutor


class RingFilter(Node):
    def __init__(self):
        super().__init__('ring_filter')
        
        # Priority levels for task management
        self.PRIORITY_EMERGENCY = 4  # Highest priority for urgent situations
        self.PRIORITY_FACE = 1       # Normal priority for face interactions
        self.PRIORITY_RING = 3       # Normal priority for ring interactions
        
        # TF2 setup for coordinate transformations
        self.tf_buffer = Buffer()  # Stores transform data
        self.tf_listener = TransformListener(self.tf_buffer, self)  # Listens for transforms
        
        # Subscribers
        self.create_subscription(
            RingMsg, 
            '/ring_coordinates',  # Input topic for detected people
            self.marker_callback, 
            10
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_pose',  # Robot localization topic
            self.pose_callback,
            10
        )
        
        # Publisher for sending tasks to main controller
        self.task_pub = self.create_publisher(Task, '/tasks', 10)
        
        # Publisher for visualization markers
        self.marker_pub = self.create_publisher(MarkerArray, 'waypoints', 10)
        self._next_marker_id = 0

        # Cluster configuration parameters
        self.cluster_radius = 1.0  # Max distance between detection positions to consider same cluster
        self.min_cluster_size = 4  # Minimum observations before publishing a task
        self.new_detection_threshold = 0.5  # Distance to consider a detection new/unique
        self.movement_threshold = 0.1  # Minimum movement to publish update (in meters)
        
        # Data storage
        self.current_pc = None  # Stores latest point cloud
        self.current_robot_pose = None  # Stores current robot pose
        
        # Cluster history now tracks both current and published states
        self.position_history = defaultdict(lambda: {
            'points': deque(maxlen=self.min_cluster_size+5),  # Stores tuples of (position, destination, robot_pose_at_detection)
            'position': None,            # Current average position
            'normal_dir': 0,         # positive=left/right, negative=up/down
            'last_published_position': None,    # Last published position
            'last_published_normal': 0  # 1 - left/right, -1 - up/down, 0 - not yet published
        })
        self.next_cluster_id = 0  # Unique ID for each cluster

        # ----------------- Marker queue for detection tracking -----------------
        self.ring_queue = deque()               # will hold (Marker, frame_id, stamp)
        self.create_timer(0.1, self._process_ring_queue)
        self.thread_executor = ThreadPoolExecutor(max_workers=1)
        
        

    def pose_callback(self, msg):
        """Store current robot pose for distance calculations"""
        # self.get_logger().info(f"Received robot pose: {msg.pose.pose.position.x}, {msg.pose.pose.position.y}, {msg.pose.pose.position.z}")
        self.current_robot_pose = msg.pose.pose

    def marker_callback(self, msg):
        if not self.current_robot_pose:
            self.get_logger().info("No robot pose available - cannot process ring detection")
            return
        
        # distance = np.linalg.norm(np.array([
        #     target_point.x,
        #     target_point.y
        # ]))
        # if distance > 1.25:
        #     self.get_logger().info(f"Face too far away ({distance:.2f}m), ignoring")
        #     return
        # if distance < 0.15:
        #     self.get_logger().info(f"Face too close ({distance:.2f}m), ignoring")
        #     return
        
        self.ring_queue.append((msg, msg.target_pose.header.frame_id, msg.target_pose.header.stamp))

    def _process_ring_queue(self):
        zero = Duration(seconds=0.0)
        last_successful_index = -1
        for i, (msg, frame_id, stamp) in enumerate(self.ring_queue):
            # wait until we can transform into map
            target_point = msg.target_pose.pose.position
            if not self.tf_buffer.can_transform('map', frame_id, stamp, zero):
                self.get_logger().info(f"Waiting for transform from {frame_id} to map")
                continue

            # transform the ring detection itself
            map_point_stamped = self.transform_to_map(target_point, frame_id, stamp)
            if map_point_stamped is None:
                continue
            world_ring = map_point_stamped.point
            
            # Also transform the normal vector to map frame
            try:
                normal = msg.normal
                tf = self.tf_buffer.lookup_transform(
                    'map',             # target frame
                    frame_id,           # source frame
                    stamp,             # when
                    timeout=Duration(seconds=0.1)
                )
                
                # Transform the normal vector (as a Vector3Stamped)
                normal_stamped = Vector3Stamped()
                normal_stamped.header.frame_id = frame_id
                normal_stamped.header.stamp = stamp
                normal_stamped.vector = normal
                map_normal = do_transform_vector3(normal_stamped, tf).vector
                
                # hand off to your handler using world-frame points and normal
                self._handle_mapped_ring(world_ring, map_normal, msg.color)
                last_successful_index = i
            except TransformException as ex:
                self.get_logger().warning(f"Failed to transform normal vector: {ex}")
                continue

        # drop processed entries
        if last_successful_index >= 0:
            self.ring_queue = deque(list(self.ring_queue)[last_successful_index+1:])
    
    def _handle_mapped_ring(self, map_point, normal, color):
        
        # self.get_logger().info(f"Detected ring at {map_point.x}, {map_point.y}, {map_point.z}")

        # Publish a sphere marker at the detection
        self.publish_to_map(map_point)

        # Compute wall normal
        normal = self.axis_align(normal, threshold_degrees=20.0)
        if normal is None:
            self.get_logger().error("Normal vector is not axis-aligned, ignoring ring detection")
            return
        
        self.publish_vector_to_map(normal, map_point)
        self.process_clusters(map_point, normal)
    
    def axis_align(self, normal, threshold_degrees=5.0):
        """
        Check if normal vector is approximately aligned with a major axis (X, Y, or Z)
        within the specified angular threshold.
        
        Args:
            normal: geometry_msgs/Vector3 - The normal vector to check
            threshold_degrees: float - Maximum angle deviation from axis (in degrees)
        
        Returns:
            bool: True if normal is axis-aligned within threshold
        """
        # Convert threshold to radians
        threshold_rad = math.radians(threshold_degrees)
        
        # Get the normal vector as numpy array
        n = np.array([normal.x, normal.y, normal.z])
        
        # Define the major axes
        axes = [
            np.array([1, 0, 0]),  # X-axis
            np.array([0, 1, 0]),  # Y-axis
            np.array([0, 0, 1]),  # Z-axis
            np.array([-1, 0, 0]),  # Negative X-axis
            np.array([0, -1, 0]),  # Negative Y-axis
            np.array([0, 0, -1])   # Negative Z-axis
        ]
        
        min_angle = float('inf')  # Initialize minimum angle
        # Check angle between normal and each axis
        for axis in axes:
            # Calculate angle between vectors
            dot_product = np.dot(n, axis)
            angle = math.acos(np.clip(dot_product / (np.linalg.norm(n) * np.linalg.norm(axis)), -1.0, 1.0))
            
            if angle < min_angle:
                min_angle = angle
            
            # If angle is within threshold of 0 or 180 degrees
            if angle < threshold_rad or (math.pi - angle) < threshold_rad:
                return axis
        
        self.get_logger().info(f"Minimum angle to axes: {math.degrees(min_angle):.2f} degrees")
        return None
        
    def publish_to_map(self, map_point, r=1.0, g=0.2, b=0.2, a=0.9):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'waypoints'
        m.id = self._next_marker_id
        self._next_marker_id += 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position = map_point
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.5
        m.color = ColorRGBA(r=r, g=g, b=b, a=a)
        m.lifetime.sec = 2
        arr = MarkerArray()
        arr.markers.append(m)
        self.marker_pub.publish(arr)
        
        # self.get_logger().info(f"Published marker at {map_point.x}, {map_point.y}, {map_point.z}")
    
    def publish_vector_to_map(self, vector, point, r=0.0, g=1.0, b=0.0, a=0.9):
        normal_marker = Marker()
        normal_marker.header.frame_id = "map"
        normal_marker.header.stamp = self.get_clock().now().to_msg()
        normal_marker.type = Marker.ARROW
        normal_marker.id = self._next_marker_id
        self._next_marker_id += 1
        normal_marker.action = Marker.ADD
        
        avg_x = point.x
        avg_y = point.y
        avg_z = point.z
        
        normal_length = 0.5  # meters
        normal_marker.points = [
            Point(x=float(avg_x), y=float(avg_y), z=float(avg_z)),
            Point(x=float(avg_x + vector[0]*normal_length),
                y=float(avg_y + vector[1]*normal_length),
                z=float(avg_z + vector[2]*normal_length))
        ]
        
        normal_marker.scale = Vector3(x=0.2, y=0.4, z=0.0)  # Shaft and head dimensions
        normal_marker.color.r = 1.0
        normal_marker.color.g = 1.0
        normal_marker.color.b = 0.0
        normal_marker.color.a = 1.0
        normal_marker.lifetime = rclpy.duration.Duration(seconds=2).to_msg()
        
        array = MarkerArray()
        array.markers.append(normal_marker)
        self.marker_pub.publish(array)

    def transform_to_map(self, point, source_frame, stamp):
        """
        Transform a Point from `source_frame` at time `stamp` into the `map` frame.
        Returns a PointStamped in the map frame, or None if the transform fails.
        """
        try:
            # Lookup the transform from source_frame → map at the right time
            tf = self.tf_buffer.lookup_transform(
                'map',             # target frame
                source_frame,      # source frame
                stamp,             # when
                timeout=Duration(seconds=0.1)
            )

            # Pack your Point into a stamped message
            ps = PointStamped()
            ps.header.frame_id = source_frame
            ps.header.stamp = stamp
            ps.point = point

            # Do the actual transform
            map_ps = do_transform_point(ps, tf)
            return map_ps

        except TransformException as ex:
            self.get_logger().warning(
                f"transform_to_map failed: cannot transform from "
                f"{source_frame} to map at {stamp}: {ex}"
            )
            return None
        

    def process_clusters(self, position, normal):
        """
        Cluster faces based on position and destination similarity
        with improved distance checking and publication tracking
        """
        if not self.current_robot_pose:
            self.get_logger().info("No robot pose available - cannot calculate detection distance")
            return

        matched_cluster = None
        
        # Calculate current detection distance (robot to ring)
        current_distance = math.sqrt(
            (position.x - self.current_robot_pose.position.x)**2 +
            (position.y - self.current_robot_pose.position.y)**2
        )
        
        # Check against all existing clusters
        for cluster_id, cluster_data in self.position_history.items():
            # Get cluster's average position and destination
            cluster_pos = cluster_data['position']
            
            # Calculate Euclidean distances
            pos_distance = np.linalg.norm(np.array([
                position.x - cluster_pos.x,
                position.y - cluster_pos.y
            ])) if cluster_pos else float('inf')
            
            # Match if within thresholds for both position and destination
            if pos_distance < self.cluster_radius:
                matched_cluster = cluster_id
                
                # Increment normal counters (Always axis aligned)
                if abs(normal[0]) > 0.5:
                    cluster_data['normal_dir'] += 1  # Left/right
                else:
                    cluster_data['normal_dir'] -= 1
                
                # If the cluster is small, add point and break
                if len(cluster_data['points']) < self.min_cluster_size:
                    # If cluster is small, add point and break
                    cluster_data['points'].append((
                        position,
                        self.current_robot_pose  # Store detection pose
                    ))
                    break
                
                # Else find the farthest point in the cluster (by detection distance, aka. how far the robot was from the detection)
                farthest_point_index = None
                farthest_distance = 0
                
                
                for i, (point, detection_pose) in enumerate(cluster_data['points']):
                    if detection_pose is None:
                        continue
                        
                    # Calculate original detection distance
                    point_distance = math.sqrt(
                        (point.x - detection_pose.position.x)**2 +
                        (point.y - detection_pose.position.y)**2
                    )
                    if point_distance > farthest_distance:
                        farthest_distance = point_distance
                        farthest_point_index = i
                
                # Replace farthest point if current is closer
                if farthest_point_index is not None and current_distance < farthest_distance:
                    cluster_data['points'][farthest_point_index] = (
                        position, 
                        self.current_robot_pose  # Store detection pose
                    )
                break

        if matched_cluster is None:
            # If no match, check if this is a genuinely new detection
            if self.is_new_emergency(position):
                # Create new cluster
                matched_cluster = self.next_cluster_id
                self.next_cluster_id += 1
                self.position_history[matched_cluster]['points'].append((
                    position,
                    self.current_robot_pose
                ))
                self.position_history[matched_cluster]['position'] = position
                
                
                # Publish emergency task to stop robot and look at detection, so that we can make sure it is not a false positive
                self.publish_emergency(position)
            return
        
        # Update cluster averages
        self.position_history[matched_cluster]['position'] = self.average_position(
            self.position_history[matched_cluster]['points'])

        # Get new averages
        new_avg_position = self.position_history[matched_cluster]['position']
        
        # Get last published averages
        last_pub_pos = self.position_history[matched_cluster]['last_published_position']
        last_pub_normal = self.position_history[matched_cluster]['last_published_normal']
        
        # Calculate movement from last published position
        position_movement = math.sqrt(
            (new_avg_position.x - last_pub_pos.x)**2 +
            (new_avg_position.y - last_pub_pos.y)**2
        ) if last_pub_pos else float('inf')
        
        # Publish if meets minimum size and significant movement
        if len(self.position_history[matched_cluster]['points']) >= self.min_cluster_size:
            if (last_pub_pos is None or  # First publication
                position_movement > self.movement_threshold or 
                last_pub_normal * self.position_history[matched_cluster]['normal_dir'] < 0):
                
                # Update last published values
                self.position_history[matched_cluster]['last_published_position'] = new_avg_position
                normal_dir = self.position_history[matched_cluster]['normal_dir']
                if normal_dir > 0:
                    normal_dir = 1 
                elif normal_dir < 0:
                    normal_dir = -1
                else:
                    self.get_logger().error("Normal direction is zero, not publishing task")
                    return
                self.publish_task(matched_cluster)
    
    def average_position(self, points):
        """Calculate mean position of all points in cluster"""
        avg = Point()
        positions = [p[0] for p in points if p[0] is not None]
        if positions:
            avg.x = np.mean([p.x for p in positions])
            avg.y = np.mean([p.y for p in positions])
        return avg

    def is_new_emergency(self, position):
        """Check if position is far enough from all known clusters to be considered new"""
        for cluster_data in self.position_history.values():
            cluster_pos = cluster_data['position']
            distance = np.linalg.norm(np.array([
                position.x - cluster_pos.x,
                position.y - cluster_pos.y
            ]))
            if distance < self.new_detection_threshold:
                return False
        return True

    def publish_emergency(self, position):
        """Create and publish high-priority emergency task"""
        task = Task()
        task.priority = self.PRIORITY_EMERGENCY
        task.task_type = "emergency"
        task.target_pose = PoseStamped()
        task.target_pose.header.frame_id = "map"
        # Current robot position is used to stop the robot
        task.target_pose.pose.position = Point(
            x=self.current_robot_pose.position.x,
            y=self.current_robot_pose.position.y,
            z=0.0  # Keep on ground plane
        )
        task.description = "Emergency stop"
        
        # after task.target_pose.pose.position = position
        dx = position.x - self.current_robot_pose.position.x
        dy = position.y - self.current_robot_pose.position.y
        yaw = math.atan2(dy, dx)
        task.target_pose.pose.orientation.z = math.sin(yaw/2)
        task.target_pose.pose.orientation.w = math.cos(yaw/2)

        self.task_pub.publish(task)
        self.get_logger().info(f"Published emergency task at {position.x}, {position.y}, {position.z}")

    def publish_task(self, cluster_id):
        """Create and publish navigation task for interacting with cluster"""
        cluster = self.position_history[cluster_id]
        if len(cluster['points']) < self.min_cluster_size:
            return

        task = Task()
        task.priority = self.PRIORITY_RING
        task.task_type = "ring"
        task.id = cluster_id
        
        # Set destination pose (1m from wall)
        task.target_pose = PoseStamped()
        task.target_pose.header.frame_id = "map"
        task.target_pose.header.stamp = self.get_clock().now().to_msg()
        task.target_pose.pose.position = cluster['position']
        
        # Calculate orientation to face the person
        task.target_pose.pose.orientation.z = 0.0
        task.target_pose.pose.orientation.w = 0.0
        
        task.description = f"Ring cluster {cluster_id}"
        self.task_pub.publish(task)
        self.get_logger().info(f"Published task for cluster {cluster_id} at {task.target_pose.pose.position.x}, {task.target_pose.pose.position.y}, {task.target_pose.pose.position.z}")


def main():
    rclpy.init()
    node = RingFilter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()