#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from collections import deque
import time
from yapper import Yapper
from enum import Enum, auto

from rclpy.qos import QoSProfile, QoSHistoryPolicy
from robot_commander import RobotCommander

class NavigationMode(Enum):
    WAYPOINTS = auto()
    FACES = auto()

class HybridController(RobotCommander):
    def __init__(self):
        super().__init__('hybrid_controller')
        
        # Navigation setup
        self.waypoints = self.get_default_waypoints()
        self.face_list = []  # Stores tuples of (face_id, pose)
        self.face_index = 0
        self.waypoint_index = 0
        self.last_waypoint_before_faces = None
        self.current_mode = NavigationMode.WAYPOINTS
        
        # Speech component
        self.yapper = Yapper()
        
        # Face detection subscription
        self.face_sub = self.create_subscription(
            Marker,
            '/filtered_people_marker',
            self.face_callback,
            10
        )

        self.get_logger().info("Hybrid controller initialized")

    def get_default_waypoints(self):
        """Define your default waypoint route here"""
        def create_pose(x, y, yaw=0.0):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation = self.YawToQuaternion(yaw)
            return pose
        
        return [
            create_pose(-0.15, -1.91),
            create_pose(3.04, -1.13),
            create_pose(2.34, 0.00),
            create_pose(2.06, 2.80),
            create_pose(-1.42, 3.26),
            create_pose(-1.48, 4.82),
            create_pose(-1.67, 1.18),
            create_pose(0.14, 1.93),
            create_pose(1.06, -0.09),
            create_pose(2.39, 0.06)
        ]

    def face_callback(self, msg):
        """Handle incoming face detections with updates"""
        face_id = msg.id
        new_pose = PoseStamped()
        new_pose.header = msg.header
        new_pose.pose = msg.pose

        # Update existing or add new face
        for idx, (f_id, pose) in enumerate(self.face_list):
            if f_id == face_id:
                self.face_list[idx] = (face_id, new_pose)
                return
        self.face_list.append((face_id, new_pose))
        self.get_logger().info(f"New face tracked: {face_id}")

    def initialize_robot(self):
        """Initialize and undock the robot"""
        self.waitUntilNav2Active()
        while self.is_docked is None:
            rclpy.spin_once(self, timeout_sec=0.5)
        if self.is_docked:
            self.undock()
            while self.is_docked:
                rclpy.spin_once(self, timeout_sec=0.5)

    def run(self):
        """Main execution loop"""
        self.initialize_robot()
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.current_mode == NavigationMode.WAYPOINTS:
                    self.process_waypoints()
                else:
                    self.process_faces()
        finally:
            self.destroyNode()
            rclpy.shutdown()

    def process_waypoints(self):
        """Process waypoints with face checking"""
        if self.waypoint_index < len(self.waypoints):
            self.last_waypoint_before_faces = self.waypoints[self.waypoint_index]
            if self.goToPose(self.waypoints[self.waypoint_index]):
                self.wait_for_task_completion(f"Waypoint {self.waypoint_index+1}")
                self.waypoint_index += 1
        else:
            self.get_logger().info("All waypoints completed, switching to face navigation")
            self.current_mode = NavigationMode.FACES
            self.face_index = 0

    def process_faces(self):
        """Process faces in detection order"""
        if self.face_index < len(self.face_list):
            face_id, face_pose = self.face_list[self.face_index]
            if self.goToPose(face_pose):
                self.wait_for_task_completion(f"Face {self.face_index+1}")
                self.execute_face_behavior(face_pose)
                self.face_index += 1
        else:
            self.get_logger().info("All faces visited")
            if self.last_waypoint_before_faces:
                self.goToPose(self.last_waypoint_before_faces)

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
        """Simple face interaction"""
        self.get_logger().info(f"Reached face at {face_pose.pose.position.x:.2f}, {face_pose.pose.position.y:.2f}")
        try:
            self.yapper.yap("Hello there!")
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()