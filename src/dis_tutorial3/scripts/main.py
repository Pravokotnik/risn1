#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from collections import deque
import time
import pyttsx3
from yapper import Yapper

from robot_commander import RobotCommander  # Your existing robot commander

class HybridController(RobotCommander):
    def __init__(self):
        super().__init__('hybrid_controller')
        
        # Navigation queues
        self.face_queue = deque()
        self.waypoints = self.get_default_waypoints()
        self.current_waypoint_index = 0
        
        # Yapper
        self.yapper = Yapper()
        
        # State management
        self.interrupted = False
        self.saved_waypoint_index = 0
        
        # ROS2 subscriptions
        self.face_sub = self.create_subscription(
            Marker,
            '/filtered_people_marker',
            self.face_callback,
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
        
        # Example waypoints (modify as needed)
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
                
                # 1. Handle face interrupts if any exist
                if self.face_queue and not self.interrupted:
                    self.handle_interrupt()
                    continue
                    
                # 2. Normal waypoint navigation
                if not self.interrupted and self.current_waypoint_index < len(self.waypoints):
                    self.navigate_to_waypoint()
                    
                # 3. Completion handling
                elif self.current_waypoint_index >= len(self.waypoints):
                    self.handle_completion()
                    
        except Exception as e:
            self.get_logger().error(f"Fatal error: {str(e)}")
        finally:
            self.get_logger().info("Shutting down")

    def handle_interrupt(self):
        """Process face detection interrupt"""
        self.get_logger().info("Face detected - interrupting current task")
        self.interrupted = True
        self.saved_waypoint_index = self.current_waypoint_index
        
        # Cancel current navigation
        self.cancelTask()
        
        # Process face
        face_pose = self.face_queue.popleft()
        if self.goToPose(face_pose):
            self.wait_for_task_completion("Approaching face")
            self.execute_face_behavior(face_pose)
        
        # Resume normal operation
        self.interrupted = False
        self.get_logger().info("Resuming normal navigation")

    def navigate_to_waypoint(self):
        """Navigate to next waypoint"""
        waypoint = self.waypoints[self.current_waypoint_index]
        if self.goToPose(waypoint):
            self.wait_for_task_completion(
                f"Navigating to waypoint {self.current_waypoint_index+1}/{len(self.waypoints)}"
            )
            self.current_waypoint_index += 1

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
        """Your custom face interaction logic"""
        self.get_logger().info(f"Executing face behavior at X:{face_pose.pose.position.x:.2f}, Y:{face_pose.pose.position.y:.2f}")
        
        # Example behaviors:
        # 1. Play sound
        try:
            # Speak directly (no file saving needed)
            self.yapper.yap("What is up my G!")
            
        except Exception as e:
            self.get_logger().error(f"TTS error: {str(e)}")
        
        # 2. Pause for interaction
        # time.sleep(1.0)
        
        # # Return to original orientation
        # self.spin(-3.14)

    def handle_completion(self):
        """Handle completion of all waypoints"""
        self.get_logger().info("All waypoints completed - waiting for new faces")
        time.sleep(5.0)  # Prevent CPU overload while idling

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