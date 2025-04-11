#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from collections import deque
import time
from yapper import Yapper
from enum import Enum, auto

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
        
        waypoints.append(create_pose(-1.07, 0.97, -0.00))
        waypoints.append(create_pose(-1.63, 4.38, 0.00))
        waypoints.append(create_pose(2.32, 2.31, -0.00))
        waypoints.append(create_pose(0.00, 1.91, -0.00))
        waypoints.append(create_pose(0.98, -0.07, -0.00))
        waypoints.append(create_pose(-0.23, -1.84, -0.00))
        waypoints.append(create_pose(-1.70, -0.52, -0.00))
        
        return waypoints

    def face_callback(self, msg):
        """Handle incoming face detections"""
        face_pose = PoseStamped()
        face_pose.header = msg.header
        face_pose.pose = msg.pose
        self.face_queue.append(face_pose)
        self.get_logger().info(f"New face detected at X:{msg.pose.position.x:.2f}, Y:{msg.pose.position.y:.2f}")

    def ring_callback(self, msg):
        """Handle incoming ring detections"""
        ring_pose = PoseStamped()
        ring_pose.header = msg.header
        ring_pose.pose = msg.pose
        ring_pose.color = msg.color
        self.ring_queue.append(ring_pose)
        self.get_logger().info(f"New ring detected at X:{msg.pose.position.x:.2f}, Y:{msg.pose.position.y:.2f}")

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
        """Process all detected rings"""
        if self.ring_queue:
            ring_pose = self.ring_queue.popleft()
            if self.goToPose(ring_pose):
                self.wait_for_task_completion("Approaching ring")
                self.execute_ring_behavior(ring_pose)
        else:
            self.get_logger().info("All rings visited, waiting for new detections")
            time.sleep(1.0)

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

    def execute_ring_behavior(self, ring_pose):
        """Custom ring interaction logic"""
        self.get_logger().info(f"Executing ring behavior at X:{ring_pose.pose.position.x:.2f}, Y:{ring_pose.pose.position.y:.2f}")
        try:
            color = ring_pose.color
            color_name = ""
            if color.r > 0.5 and color.g < 0.5 and color.b < 0.5:
                color_name = "red"
            elif color.r < 0.5 and color.g > 0.5 and color.b < 0.5:
                color_name = "green"
            elif color.r < 0.5 and color.g < 0.5 and color.b > 0.5:
                color_name = "blue"
            elif color.r < 0.5 and color.g < 0.5 and color.b < 0.5:
                color_name = "black"
            else:
                color_name = "unknown"
            self.yapper.yap(f"{color_name}")
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