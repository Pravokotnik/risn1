#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from collections import deque
import time
from yapper import Yapper
from enum import Enum, auto
import numpy as np

from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy

from robot_commander import RobotCommander

class NavigationMode(Enum):
    WAYPOINTS = auto()
    FACES = auto()
    RINGS = auto()

class HybridController(RobotCommander):
    def __init__(self):
        super().__init__('hybrid_controller')
        
        # Navigation queues
        self.waypoints = self.get_default_waypoints()
        self.face_queue = deque()
        self.ring_queue = deque()
        
        # Ring tracking dictionary
        # Structure: {ring_id: {'pose': pose, 'colors': {color_name: count}}}
        self.rings_dict = {}
        
        # Current navigation state
        self.current_mode = NavigationMode.WAYPOINTS
        self.current_index = 0
        self.interrupted = False
        
        # Yapper for speech
        self.yapper = Yapper()
        
        # ROS2 subscriptions
        self.face_sub = self.create_subscription(
            Marker,
            '/filtered_people_marker',
            self.face_callback,
            10
        )
        
        self.ring_sub = self.create_subscription(
            Marker,
            '/filtered_rings_marker',
            self.ring_callback,
            10
        )

        # za markerje
        qos = QoSProfile(
            depth=5,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        self.marker_publisher = self.create_publisher(MarkerArray, '/waypoints', qos)
        
        self.get_logger().info("Hybrid controller initialized")

    def get_default_waypoints(self):
        """Define your default waypoint route here"""
        waypoints = []
        
        def create_pose(x, y, yaw=0.0):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation = self.YawToQuaternion(yaw)
            return pose
        
        #waypoints.append(create_pose(-1.07, 0.97, -0.00))
        #waypoints.append(create_pose(-1.63, 4.38, 0.00))
        #waypoints.append(create_pose(2.32, 2.31, -0.00))
        #waypoints.append(create_pose(0.00, 1.91, -0.00))
        #waypoints.append(create_pose(0.98, -0.07, -0.00))
        #waypoints.append(create_pose(-0.23, -1.84, -0.00))
        #waypoints.append(create_pose(-1.70, -0.52, -0.00))

        waypoints.append(create_pose(-0.15, -1.91, -0.00))
        waypoints.append(create_pose(3.04, -1.13, -0.00))
        waypoints.append(create_pose(2.34, 0.00, 0.01))
        waypoints.append(create_pose(2.06, 2.80, 0.01))
        waypoints.append(create_pose(-1.42, 3.26, -0.00))
        waypoints.append(create_pose(-1.48, 4.82, 0.00))
        waypoints.append(create_pose(-1.67, 1.18, 0.01))
        waypoints.append(create_pose(0.14, 1.93, 0.01))
        waypoints.append(create_pose(1.06, -0.09, 0.01))
        waypoints.append(create_pose(2.39, 0.06, 0.01))
        
        return waypoints

    def face_callback(self, msg):
        """Handle incoming face detections"""
        face_pose = PoseStamped()
        face_pose.header = msg.header
        face_pose.pose = msg.pose
        self.face_queue.append(face_pose)
        self.get_logger().info(f"New face detected at X:{msg.pose.position.x:.2f}, Y:{msg.pose.position.y:.2f}")

    def ring_callback(self, msg):
        """Handle incoming ring detections with their ID"""
        ring_pose = PoseStamped()
        ring_pose.header = msg.header
        ring_pose.pose = msg.pose
        # Get the ring ID from the marker
        ring_id = msg.id
        # Determine color from the marker's color field
        color = self.determine_color_name(msg.color)
        
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.id = ring_id
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        
        # Set position for the marker (same for both new and updated)
        marker.pose = msg.pose
        
        # Store or update the ring in our dictionary
        if ring_id not in self.rings_dict:
            self.rings_dict[ring_id] = {
                'pose': ring_pose,
                'colors': {color: 1}
            }
            self.get_logger().info(f"New ring {ring_id} detected: {color} at X:{msg.pose.position.x:.2f}, Y:{msg.pose.position.y:.2f}")
            
            if color == "red":
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif color == "green":
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            elif color == "blue":
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            elif color == "black":  # Assuming black is a valid color
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            
            # Add to navigation queue (only add new rings)
            self.ring_queue.append(ring_id)
        else:
            # Update existing ring
            self.rings_dict[ring_id]['pose'] = ring_pose  # Update position
            
            # Update color count
            if color in self.rings_dict[ring_id]['colors']:
                self.rings_dict[ring_id]['colors'][color] += 1
            else:
                self.rings_dict[ring_id]['colors'][color] = 1
                
            # Determine most frequent color for display
            most_common_color = max(self.rings_dict[ring_id]['colors'], 
                                key=self.rings_dict[ring_id]['colors'].get)
            
            # Set color based on most common detection
            if most_common_color == "red":
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif most_common_color == "green":
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            elif most_common_color == "blue":
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            elif most_common_color == "black":
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                
            self.get_logger().info(f"Updated ring {ring_id}: most common color is {most_common_color} at X:{msg.pose.position.x:.2f}, Y:{msg.pose.position.y:.2f}")
        
        # Add marker to array and publish (for both new and updated markers)
        marker_array.markers.append(marker)
        self.marker_publisher.publish(marker_array)

    def determine_color_name(self, color_msg):
        """Convert ROS color message to color name"""
        if color_msg.r > 0.5 and color_msg.g < 0.5 and color_msg.b < 0.5:
            return "red"
        elif color_msg.r < 0.5 and color_msg.g > 0.5 and color_msg.b < 0.5:
            return "green"
        elif color_msg.r < 0.5 and color_msg.g < 0.5 and color_msg.b > 0.5:
            return "blue"
        elif color_msg.r < 0.2 and color_msg.g < 0.2 and color_msg.b < 0.2:
            return "black"
        else:
            return "unknown"

    def get_most_common_color(self, ring_id):
        """Get the most commonly detected color for a ring"""
        if ring_id not in self.rings_dict:
            return "unknown"
            
        colors = self.rings_dict[ring_id]['colors']
        if not colors:
            return "unknown"
            
        # Find color with highest count
        most_common_color = max(colors.items(), key=lambda x: x[1])
        return most_common_color[0]

    def initialize_robot(self):
        """Initialize and undock the robot"""
        self.waitUntilNav2Active()
        
        # Wait for dock status
        while self.is_docked is None:
            self.get_logger().info("Waiting for dock status...", throttle_duration_sec=5)
            rclpy.spin_once(self, timeout_sec=0.5)
        
        # Undock if needed
        if self.is_docked:
            self.get_logger().info("Undocking...")
            self.undock()
            while self.is_docked:  # Wait until undocking completes
                rclpy.spin_once(self, timeout_sec=0.5)
            self.get_logger().info("Undocking complete")

    def run(self):
        """Main execution loop"""
        self.initialize_robot()
        
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)  # Process callbacks
                
                if not self.interrupted:
                    if self.current_mode == NavigationMode.WAYPOINTS:
                        self.process_waypoints()
                    elif self.current_mode == NavigationMode.FACES:
                        self.process_faces()
                    elif self.current_mode == NavigationMode.RINGS:
                        self.process_rings()
                    
        except Exception as e:
            self.get_logger().error(f"Fatal error: {str(e)}")
        finally:
            self.get_logger().info("Shutting down")

    def process_waypoints(self):
        """Process all waypoints before moving to next mode"""
        if self.current_index < len(self.waypoints):
            waypoint = self.waypoints[self.current_index]
            if self.goToPose(waypoint):
                self.wait_for_task_completion(
                    f"Navigating to waypoint {self.current_index+1}/{len(self.waypoints)}"
                )
                self.current_index += 1
        else:
            self.get_logger().info("All waypoints completed, switching to face navigation")
            self.current_mode = NavigationMode.FACES
            self.current_index = 0

    def process_faces(self):
        """Process all detected faces before moving to ring detection"""
        if self.face_queue:
            face_pose = self.face_queue.popleft()
            if self.goToPose(face_pose):
                self.wait_for_task_completion("Approaching face")
                self.execute_face_behavior(face_pose)
        else:
            self.get_logger().info("All faces visited, switching to ring detection")
            self.current_mode = NavigationMode.RINGS

    def process_rings(self):
        """Process all detected rings with improved color accuracy"""
        # First, filter the rings if we have more than 4 detections
        if len(self.rings_dict) > 4:
            self.clean_up_rings()
        
        if self.ring_queue:
            ring_id = self.ring_queue.popleft()
            
            # Skip if the ring has been removed from the dictionary
            if ring_id not in self.rings_dict:
                return
                
            ring_pose = self.rings_dict[ring_id]['pose']
            
            if self.goToPose(ring_pose):
                self.wait_for_task_completion("Approaching ring")
                
                # Get the most common color for this ring
                color = self.get_most_common_color(ring_id)
                
                self.execute_ring_behavior(ring_id, ring_pose, color)
        else:
            self.get_logger().info("All rings visited, waiting for new detections")
            time.sleep(1.0)

    def clean_up_rings(self):
        """Filter the rings to keep only the 4 most distinct and confident detections"""
        # Skip if we don't have enough rings
        if len(self.rings_dict) <= 4:
            return
        
        self.get_logger().info(f"Cleaning up rings: {len(self.rings_dict)} rings detected, keeping best 4")
        
        # Step 1: Calculate confidence scores for each ring
        # Score = total number of color detections
        ring_scores = {}
        for ring_id, ring_data in self.rings_dict.items():
            total_detections = sum(ring_data['colors'].values())
            ring_scores[ring_id] = total_detections
        
        # Step 2: Identify clusters of rings that are close to each other
        clusters = []
        processed_rings = set()
        
        distance_threshold = 1.0  # Adjust as needed based on your environment
        
        for ring_id in self.rings_dict:
            if ring_id in processed_rings:
                continue
            
            # Start a new cluster
            cluster = [ring_id]
            processed_rings.add(ring_id)
            
            # Get ring position
            pos1 = self.rings_dict[ring_id]['pose'].pose.position
            pos1_array = np.array([pos1.x, pos1.y, pos1.z])
            
            # Find other rings close to this one
            for other_id in self.rings_dict:
                if other_id in processed_rings:
                    continue
                    
                pos2 = self.rings_dict[other_id]['pose'].pose.position
                pos2_array = np.array([pos2.x, pos2.y, pos2.z])
                
                # Check distance
                if np.linalg.norm(pos1_array - pos2_array) < distance_threshold:
                    cluster.append(other_id)
                    processed_rings.add(other_id)
            
            clusters.append(cluster)
        
        # Step 3: For each cluster, keep only the ring with highest score
        rings_to_keep = []
        for cluster in clusters:
            if len(cluster) == 1:
                rings_to_keep.append(cluster[0])
            else:
                # Find ring with highest score in this cluster
                best_ring = max(cluster, key=lambda ring_id: ring_scores[ring_id])
                rings_to_keep.append(best_ring)
                
                # Log the merge action
                self.get_logger().info(f"Merged rings {cluster} into {best_ring} (score: {ring_scores[best_ring]})")
        
        # Step 4: If we still have more than 4 rings, keep the 4 with highest scores
        if len(rings_to_keep) > 4:
            rings_to_keep.sort(key=lambda ring_id: ring_scores[ring_id], reverse=True)
            rings_to_keep = rings_to_keep[:4]
        
        # Step 5: Remove rings that weren't selected
        rings_to_remove = set(self.rings_dict.keys()) - set(rings_to_keep)
        for ring_id in rings_to_remove:
            self.get_logger().info(f"Removing ring {ring_id} (score: {ring_scores[ring_id]})")
            
            # Remove from dictionary
            del self.rings_dict[ring_id]
            
            # Remove from queue if present
            if ring_id in self.ring_queue:
                self.ring_queue.remove(ring_id)
        
        self.get_logger().info(f"Cleanup complete. Keeping rings: {rings_to_keep}")

    def wait_for_task_completion(self, task_name=""):
        """Helper for waiting on navigation tasks"""
        while not self.isTaskComplete():
            self.get_logger().info(
                f"{task_name}... Current position: "
                f"X:{self.current_pose.pose.position.x:.2f}, "
                f"Y:{self.current_pose.pose.position.y:.2f}",
                throttle_duration_sec=2.0
            )
            time.sleep(0.1)

    def execute_face_behavior(self, face_pose):
        """Custom face interaction logic"""
        self.get_logger().info(f"Executing face behavior at X:{face_pose.pose.position.x:.2f}, Y:{face_pose.pose.position.y:.2f}")
        try:
            self.yapper.yap("What is up my G!")
        except Exception as e:
            self.get_logger().error(f"TTS error: {str(e)}")

    def execute_ring_behavior(self, ring_id, ring_pose, color):
        """Custom ring interaction logic with most likely color"""
        self.get_logger().info(
            f"Executing ring behavior for ring {ring_id} at "
            f"X:{ring_pose.pose.position.x:.2f}, Y:{ring_pose.pose.position.y:.2f} "
            f"with most likely color: {color}"
        )
        
        try:
            # Get color occurrence statistics
            color_counts = self.rings_dict[ring_id]['colors']
            total_detections = sum(color_counts.values())
            confidence = color_counts.get(color, 0) / total_detections if total_detections > 0 else 0
            
            # Speak the color with confidence information
            if color != "unknown":
                message = f"I found a {color} ring! I'm {confidence*100:.0f}% confident about this color."
                self.yapper.yap(message)
            else:
                self.yapper.yap("I found a ring but I'm not sure about its color.")
        except Exception as e:
            self.get_logger().error(f"TTS error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    controller = HybridController()
    
    try:
        controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroyNode()
        rclpy.shutdown()

if __name__ == '__main__':
    main()