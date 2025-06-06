#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, PoseStamped, Point, PoseWithCovarianceStamped
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import ColorRGBA
import numpy as np
from collections import deque, defaultdict
import math
import time
from dis_tutorial3.msg import Task, FaceMsg
from tf2_geometry_msgs import do_transform_point

from message_filters import Subscriber
from std_msgs.msg import Header  # Add this import at the top of your file

from tf2_msgs.msg import TFMessage
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from concurrent.futures import ThreadPoolExecutor


class FaceFilter(Node):
    def __init__(self):
        super().__init__('face_filter')
        
        # Priority levels for task management
        self.PRIORITY_EMERGENCY = 10  # Highest priority for urgent situations
        self.PRIORITY_FACE = 2        # Normal priority for face interactions
        
        # TF2 setup for coordinate transformations
        self.tf_buffer = Buffer()  # Stores transform data
        self.tf_listener = TransformListener(self.tf_buffer, self)  # Listens for transforms
        
        # Subscribers
        self.create_subscription(
            FaceMsg, 
            '/face_coordinates',  # Input topic for detected people
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
        self.cluster_radius = 1.0  # Max distance between face positions to consider same cluster
        self.dest_radius = 1.0     # Max distance between destinations to consider same cluster
        self.min_cluster_size = 15  # Minimum observations before publishing a task
        self.new_face_threshold = 0.5  # Distance to consider a face new/unique
        self.movement_threshold = 0.1  # Minimum movement to publish update (in meters)
        
        # Data storage
        self.current_pc = None  # Stores latest point cloud
        self.current_robot_pose = None  # Stores current robot pose
        
        # Cluster history now tracks both current and published states
        self.position_history = defaultdict(lambda: {
            'points': deque(maxlen=self.min_cluster_size+5),  # Stores tuples of (position, destination, robot_pose_at_detection)
            'position': None,            # Current average position
            'destination': None,         # Current average destination
            'last_published_position': None,    # Last published position
            'last_published_destination': None  # Last published destination
        })
        self.next_cluster_id = 0  # Unique ID for each cluster

        # ----------------- Marker queue for face tracking -----------------
        self.face_queue = deque()               # will hold (Marker, frame_id, stamp)
        self.create_timer(0.1, self._process_face_queue)
        self.thread_executor = ThreadPoolExecutor(max_workers=1)
        
        

    def pose_callback(self, msg):
        """Store current robot pose for distance calculations"""
        # self.get_logger().info(f"Received robot pose: {msg.pose.pose.position.x}, {msg.pose.pose.position.y}, {msg.pose.pose.position.z}")
        self.current_robot_pose = msg.pose.pose

    def marker_callback(self, msg):
        target_point = msg.target_pose.pose.position
        if not self.current_robot_pose:
            return
        distance = np.linalg.norm(np.array([
            target_point.x,
            target_point.y
        ]))
        if distance > 2.0:
            self.get_logger().info(f"Face too far away ({distance:.2f}m), ignoring")
            return
        if distance < 0.15:
            self.get_logger().info(f"Face too close ({distance:.2f}m), ignoring")
            return
        self.get_logger().info(f"Received face detection at {target_point.x}, {target_point.y}, {target_point.z} with distance {distance:.2f}m")
        
        
        self.face_queue.append((msg, msg.target_pose.header.frame_id, msg.target_pose.header.stamp))

    def _process_face_queue(self):
        zero = Duration(seconds=0.0)
        last_successful_index = -1
        for i, (msg, frame_id, stamp) in enumerate(self.face_queue):
            # wait until we can transform into map
            target_point = msg.target_pose.pose.position
            bl_point = msg.bottom_left_point
            br_point = msg.bottom_right_point
            tl_point = msg.top_left_point
            tp_point = msg.top_right_point
            if not self.tf_buffer.can_transform('map', frame_id, stamp, zero):
                self.get_logger().info(f"Waiting for transform from {frame_id} to map")
                continue

            # transform the face detection itself
            map_point_stamped = self.transform_to_map(target_point, frame_id, stamp)
            if map_point_stamped is None:
                continue
            world_face = map_point_stamped.point
            
            
            bl_point_stamped = self.transform_to_map(bl_point, frame_id, stamp)
            if bl_point_stamped is None:
                self.get_logger().error("Failed to transform bottom left point")
                continue
            
            br_point_stamped = self.transform_to_map(br_point, frame_id, stamp)
            if br_point_stamped is None:
                self.get_logger().error("Failed to transform bottom right point")
                continue
            
            tl_point_stamped = self.transform_to_map(tl_point, frame_id, stamp)
            if tl_point_stamped is None:
                self.get_logger().error("Failed to transform top left point")
                continue
            
            tr_point_stamped = self.transform_to_map(tp_point, frame_id, stamp)
            if tr_point_stamped is None:
                self.get_logger().error("Failed to transform top point")
                continue


            # hand off to your handler using world‐frame points
            self._handle_mapped_face(world_face, 
                                     bl_point_stamped.point, 
                                     br_point_stamped.point, 
                                     tl_point_stamped.point,
                                     tr_point_stamped.point)
            last_successful_index = i

        # drop processed entries
        if last_successful_index >= 0:
            self.face_queue = deque(list(self.face_queue)[last_successful_index+1:])
    
    def _handle_mapped_face(self, map_point, bottom_left_point, bottom_right_point, top_left_point, top_right_point):
        # Transform face point into map frame
        if not map_point:
            return
        
        self.get_logger().info(f"Detected face at {map_point.x}, {map_point.y}, {map_point.z}")

        # Publish a sphere marker at the detection
        self.publish_to_map(map_point)

        # Compute wall normal
        self.get_logger().warning("Calculating wall normal...")
        normal = self.calculate_wall_normal(bottom_left_point, bottom_right_point, top_left_point)
        if normal is None:
            return

        # Compute destination off the wall
        destination = self.calculate_perpendicular_destination(map_point, normal)
        if destination is None:
            self.get_logger().error("Failed to calculate destination from wall normal.")
            return
        
        # Get distance between map point and destination. If smaller than 0.5, do no process further
        distance = np.linalg.norm(np.array([
            map_point.x - destination.x,
            map_point.y - destination.y
        ]))
        if distance < 0.4:
            self.get_logger().warn(f"Destination too close to face ({distance:.2f}m), ignoring")
            return
        
        self.publish_to_map(destination, r=0.0, g=1.0, b=0.0, a=0.9)
        self.get_logger().error(f"Calculated destination at {destination.x}, {destination.y}, {destination.z}")
        self.process_clusters(map_point, destination)
        
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
        
        self.get_logger().info(f"Published marker at {map_point.x}, {map_point.y}, {map_point.z}")

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
        

    def calculate_wall_normal(self, bottom_left_point, bottom_right_point, top_left_point):
        """
        Get vector from left to right point and get perpendicular vector by multiplying with top_point
        """
        
        # Convert points to numpy arrays for easier calculations
        bl = np.array([bottom_left_point.x, bottom_left_point.y, bottom_left_point.z])
        br = np.array([bottom_right_point.x, bottom_right_point.y, bottom_right_point.z])
        tl = np.array([top_left_point.x, top_left_point.y, top_left_point.z])
        
        # Calculate vectors along the wall
        left_to_right = br - bl  # Vector from bottom left to bottom right
        top_to_bottom = tl - bl  # Vector from bottom left to top left
        
        # Calculate the normal vector using cross product
        normal = np.cross(left_to_right, top_to_bottom)
        self.get_logger().info(f"Calculated normal vector: {normal}")
        
        # Normalize the normal vector
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            self.get_logger().error("Calculated normal vector has zero length, cannot normalize.")
            return None
        normal = normal / norm
        
        return normal

    def calculate_perpendicular_destination(self, face_point, normal):
        """Calculate stopping position 1m from wall along normal vector"""
        dest = Point()
        dest.x = face_point.x + normal[0] * 0.5  # Move 1m along normal
        dest.y = face_point.y + normal[1] * 0.5
        dest.z = 0.0  # Keep on ground plane
        return dest

    def process_clusters(self, position, destination):
        """
        Cluster faces based on position and destination similarity
        with improved distance checking and publication tracking
        """
        if not self.current_robot_pose:
            self.get_logger().info("No robot pose available - cannot calculate detection distance")
            return

        matched_cluster = None
        
        # Calculate current detection distance (robot to face)
        current_distance = math.sqrt(
            (position.x - self.current_robot_pose.position.x)**2 +
            (position.y - self.current_robot_pose.position.y)**2
        )
        
        # Check against all existing clusters
        for cluster_id, cluster_data in self.position_history.items():
            # Get cluster's average position and destination
            cluster_pos = cluster_data['position']
            cluster_dest = cluster_data['destination']
            
            # Calculate Euclidean distances
            pos_distance = np.linalg.norm(np.array([
                position.x - cluster_pos.x,
                position.y - cluster_pos.y
            ])) if cluster_pos else float('inf')
            
            dest_distance = np.linalg.norm(np.array([
                destination.x - cluster_dest.x,
                destination.y - cluster_dest.y
            ])) if cluster_dest else float('inf')
            
            # Match if within thresholds for both position and destination
            if pos_distance < self.cluster_radius and dest_distance < self.dest_radius:
                matched_cluster = cluster_id
                
                # If the cluster is small, add point and break
                if len(cluster_data['points']) < self.min_cluster_size:
                    # If cluster is small, add point and break
                    cluster_data['points'].append((
                        position,
                        destination,
                        self.current_robot_pose  # Store detection pose
                    ))
                    break
                
                # Else find the farthest point in the cluster (by detection distance, aka. how far the robot was from the face)
                farthest_point_index = None
                farthest_distance = 0
                
                for i, (point, _, detection_pose) in enumerate(cluster_data['points']):
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
                        destination,
                        self.current_robot_pose  # Store detection pose
                    )
                break

        if matched_cluster is None:
            # If no match, check if this is a genuinely new face
            if self.is_new_emergency(position):
                # Create new cluster
                matched_cluster = self.next_cluster_id
                self.next_cluster_id += 1
                self.position_history[matched_cluster]['points'].append((
                    position,
                    destination,
                    self.current_robot_pose
                ))
                self.position_history[matched_cluster]['position'] = position
                self.position_history[matched_cluster]['destination'] = destination
                
                # Publish emergency task to stop robot and look at face, so that we can make sure it is not a false positive
                self.publish_emergency(position)
            return
        
        # Update cluster averages
        self.position_history[matched_cluster]['position'] = self.average_position(
            self.position_history[matched_cluster]['points'])
        self.position_history[matched_cluster]['destination'] = self.average_destination(
            self.position_history[matched_cluster]['points'])

        # Get new averages
        new_avg_position = self.position_history[matched_cluster]['position']
        new_avg_destination = self.position_history[matched_cluster]['destination']
        
        # Get last published averages
        last_pub_pos = self.position_history[matched_cluster]['last_published_position']
        last_pub_dest = self.position_history[matched_cluster]['last_published_destination']
        
        # Calculate movement from last published position
        position_movement = math.sqrt(
            (new_avg_position.x - last_pub_pos.x)**2 +
            (new_avg_position.y - last_pub_pos.y)**2
        ) if last_pub_pos else float('inf')
        
        destination_movement = math.sqrt(
            (new_avg_destination.x - last_pub_dest.x)**2 +
            (new_avg_destination.y - last_pub_dest.y)**2
        ) if last_pub_dest else float('inf')
        
        # Publish if meets minimum size and significant movement
        if len(self.position_history[matched_cluster]['points']) >= self.min_cluster_size:
            if (last_pub_pos is None or  # First publication
                position_movement > self.movement_threshold or
                destination_movement > self.movement_threshold):
                
                self.publish_task(matched_cluster)
                # Update last published values
                self.position_history[matched_cluster]['last_published_position'] = new_avg_position
                self.position_history[matched_cluster]['last_published_destination'] = new_avg_destination
    
    def average_position(self, points):
        """Calculate mean position of all points in cluster"""
        avg = Point()
        positions = [p[0] for p in points if p[0] is not None]
        if positions:
            avg.x = np.mean([p.x for p in positions])
            avg.y = np.mean([p.y for p in positions])
        return avg

    def average_destination(self, points):
        """Calculate mean destination of all points in cluster"""
        avg = Point()
        destinations = [p[1] for p in points if p[1] is not None]
        if destinations:
            avg.x = np.mean([p.x for p in destinations])
            avg.y = np.mean([p.y for p in destinations])
        return avg


    def is_new_emergency(self, position):
        """Check if position is far enough from all known clusters to be considered new"""
        for cluster_data in self.position_history.values():
            cluster_pos = cluster_data['position']
            distance = np.linalg.norm(np.array([
                position.x - cluster_pos.x,
                position.y - cluster_pos.y
            ]))
            if distance < self.new_face_threshold:
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

    def publish_task(self, cluster_id):
        """Create and publish navigation task for interacting with face cluster"""
        cluster = self.position_history[cluster_id]
        if len(cluster['points']) < self.min_cluster_size:
            return

        task = Task()
        task.priority = self.PRIORITY_FACE
        task.task_type = "face"
        task.id = cluster_id
        
        # Set destination pose (1m from wall)
        task.target_pose = PoseStamped()
        task.target_pose.header.frame_id = "map"
        task.target_pose.header.stamp = self.get_clock().now().to_msg()
        task.target_pose.pose.position = cluster['destination']
        
        # Calculate orientation to face the person
        dx = cluster['position'].x - cluster['destination'].x
        dy = cluster['position'].y - cluster['destination'].y
        yaw = math.atan2(dy, dx)  # Angle from destination to face
        
        # Convert yaw to quaternion
        task.target_pose.pose.orientation.z = math.sin(yaw/2)
        task.target_pose.pose.orientation.w = math.cos(yaw/2)
        
        task.description = f"Face cluster {cluster_id}"
        self.task_pub.publish(task)


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