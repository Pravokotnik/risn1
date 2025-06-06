#!/usr/bin/python3

import rclpy
import rclpy.duration
from rclpy.node import Node
import os
import cv2
import datetime
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from action_msgs.msg import GoalStatus

from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

qos_profile = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1)

class ArmMoverAction(Node):
    def __init__(self):
        super().__init__('transform_point')

        # Basic ROS stuff
        timer_frequency = 1
        timer_period = 1/timer_frequency

        # General variables for setting the arm position
        self.new_command_arrived = False
        self.executing_command = False

        # Subscribers / Publishers
        self.arm_command_sub = self.create_subscription(String, "/arm_command", self.arm_command_callback, 1)
        self.arm_position_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')

        # Camera setup
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/top_camera/rgb/preview/image_raw',
            self.image_callback,
            qos_profile=QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                depth=1
            )
        )
        self.latest_image = None
        self.photo_commands = {'look_for_bird'}
        
        # Create images directory
        self.image_dir = os.path.join(os.getcwd(), 'images')
        os.makedirs(self.image_dir, exist_ok=True)

        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Predefined positions for the robot arm
        self.joint_names = ['arm_base_joint', 'arm_shoulder_joint', 'arm_elbow_joint', 'arm_wrist_joint']
        self.arm_poses = {'look_for_parking':[0.,0.4,1.5,1.2],
                          'look_for_qr':[0.,0.6,0.5,2.0],
                          'garage':[0.,-0.45,2.8,-0.8],
                          'up':[0.,0.,0.,0.],
                          'look_for_bird':[0.0, 0.0, 1.2, 0.0],
                          'manual':None}

        self.get_logger().info(f"Initialized the Arm Mover node with camera support! Images will save to: {self.image_dir}")

    def image_callback(self, msg):
        """Store the latest camera image"""
        self.latest_image = msg
        
        try:
            # Convert ROS Image to OpenCV image (BGR8)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        # Display the image in a window named "Camera"
        cv2.imshow('Camera', cv_image)

        # Necessary to allow OpenCV to process its window events
        cv2.waitKey(1)

    def capture_image(self, command_name):
        """Capture and save current camera image with timestamp"""
        if self.latest_image is None:
            self.get_logger().warn("No image received from camera yet!")
            return

        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
            
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{command_name}_{timestamp}.png"
            filepath = os.path.join(self.image_dir, filename)
            
            # Save image in highest quality
            cv2.imwrite(filepath, cv_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            self.get_logger().info(f"Saved high-quality image to {filepath}")
            
            # Also save a JPEG version with quality setting
            jpg_path = os.path.join(self.image_dir, f"{command_name}_{timestamp}.jpg")
            cv2.imwrite(jpg_path, cv_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
        except Exception as e:
            self.get_logger().error(f"Failed to capture image: {str(e)}")

    def set_arm_position(self, command_string):
        self.executing_command = True

        while not self.arm_position_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("'Arm controller' action server not available, waiting...")

        point = JointTrajectoryPoint()

        command = self.arm_poses[command_string.split(':')[0]]
        if command is None:
            self.get_logger().info(f"Received command MANUAL command {command_string.split(':')[1]}")
            point.positions = eval(command_string.split(':')[1])
        else:
            point.positions = command
        point.time_from_start = rclpy.duration.Duration(seconds=3.).to_msg()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.goal_time_tolerance = rclpy.duration.Duration(seconds=3.).to_msg()
        goal_msg.trajectory.joint_names = self.joint_names
        goal_msg.trajectory.points.append(point)

        self.get_logger().info(f'Sending a goal to the action server, position is {command}')
        self.send_goal_future = self.arm_position_client.send_goal_async(goal_msg)
        self.send_goal_future.add_done_callback(self.goal_accepted_callback)

        self.new_command_arrived = False
    
    def goal_accepted_callback(self, future):
        goal_handle = future.result()

        if goal_handle.accepted: 
            self.get_logger().info('Arm controller ACCEPTED the goal.')
            self.result_future = goal_handle.get_result_async()
            self.result_future.add_done_callback(self.get_result_callback)
        else:
            self.get_logger().error('Arm controller REJECTED the goal.')
            self.executing_command = False

    def get_result_callback(self, future):
        status = future.result().status
        
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Arm controller says GOAL FAILED: {status}')
        else:
            self.get_logger().info(f'Arm controller says GOAL REACHED.')
            # Capture image after reaching certain positions
            base_command = self.current_command.split(':')[0]
            if base_command in self.photo_commands:
                self.capture_image(base_command)
        
        self.executing_command = False

    def timer_callback(self):
        if self.new_command_arrived and not self.executing_command:
            self.set_arm_position(self.current_command)
            self.get_logger().info(f"Will set a new position for the arm joints: {self.current_command}")
            self.previous_command = self.current_command
            self.new_command_arrived = False

    def arm_command_callback(self, msg):
        command_string = msg.data.strip().lower()
        command_test = msg.data.strip().lower().split(":")[0]

        assert command_test in list(self.arm_poses.keys())

        self.current_command = command_string
        self.new_command_arrived = True
        self.get_logger().info(f"Got a new command for the arm configuration: {command_string}")

def main():
    rclpy.init(args=None)
    rd_node = ArmMoverAction()
    rclpy.spin(rd_node)

if __name__ == '__main__':
    main()