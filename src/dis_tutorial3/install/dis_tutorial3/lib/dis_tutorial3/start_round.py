#!/usr/bin/env python3

import time
from enum import Enum

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
import tf_transformations

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus
import math

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)


class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3


amcl_pose_qos = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class RobotCommander(Node):
    def __init__(self, node_name="robot_commander", namespace=""):
        super().__init__(node_name=node_name, namespace=namespace)

        self.pose_frame_id = "map"

        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.current_pose = None
        self.is_docked = None

        # ROS2 subscribers
        self.create_subscription(DockStatus, "dock_status", self._dockCallback, qos_profile_sensor_data)
        self.localization_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "amcl_pose", self._amclPoseCallback, amcl_pose_qos
        )

        # ROS2 publishers
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, "initialpose", 10)

        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.spin_client = ActionClient(self, Spin, "spin")
        self.undock_action_client = ActionClient(self, Undock, "undock")
        self.dock_action_client = ActionClient(self, Dock, "dock")

        self.get_logger().info("Robot commander has been initialized!")

    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()

    def goToPose(self, pose, behavior_tree=""):
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        self.info(
            "Navigating to goal: "
            + str(pose.pose.position.x)
            + " "
            + str(pose.pose.position.y)
            + "..."
        )
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg, self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error(
                "Goal to "
                + str(pose.pose.position.x)
                + " "
                + str(pose.pose.position.y)
                + " was rejected!"
            )
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def spin(self, spin_dist=1.57, time_allowance=10):
        self.debug("Waiting for 'Spin' action server")
        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.info("'Spin' action server not available, waiting...")
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = spin_dist
        goal_msg.time_allowance = Duration(sec=time_allowance)

        self.info(f"Spinning to angle {goal_msg.target_yaw}....")
        send_goal_future = self.spin_client.send_goal_async(goal_msg, self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error("Spin request was rejected!")
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def undock(self):
        self.info("Sending undock goal...")
        self.undock_send_goal()

        while not self.isUndockComplete():
            self.info("Waiting for undock to complete...")
            rclpy.spin_once(self, timeout_sec=0.5)
        self.info("Undock complete.")

        # Wait for a new AMCL pose after undocking
        self.info("Waiting for AMCL to re-localize after undocking...")
        self.initial_pose_received = False
        retries = 0
        while not self.initial_pose_received and retries < 20:
            rclpy.spin_once(self, timeout_sec=0.5)
            retries += 1

        if self.initial_pose_received:
            self.info("AMCL pose received after undocking!")
        else:
            self.warn("No new AMCL pose after undocking. Continuing anyway.")


    def undock_send_goal(self):
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        goal_future = self.undock_action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, goal_future)

        self.undock_goal_handle = goal_future.result()

        if not self.undock_goal_handle.accepted:
            self.error("Undock goal rejected")
            return

        self.undock_result_future = self.undock_goal_handle.get_result_async()

    def isUndockComplete(self):
        if self.undock_result_future is None or not self.undock_result_future:
            return True

        rclpy.spin_until_future_complete(self, self.undock_result_future, timeout_sec=0.1)

        if self.undock_result_future.result():
            self.undock_status = self.undock_result_future.result().status
            if self.undock_status != GoalStatus.STATUS_SUCCEEDED:
                self.info(f"Goal with failed with status code: {self.status}")
                return True
        else:
            return False

        self.info("Undock succeeded")
        return True

    def cancelTask(self):
        self.info("Canceling current task.")
        if self.result_future:
            future = self.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future)
        return

    def isTaskComplete(self):
        if not self.result_future:
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.debug(f"Task failed with status code: {self.status}")
                return True
        else:
            return False

        self.debug("Task succeeded!")
        return True

    def getFeedback(self):
        return self.feedback

    def getResult(self):
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN

    def waitUntilNav2Active(self, navigator="bt_navigator", localizer="amcl"):
        self._waitForNodeToActivate(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._waitForNodeToActivate(navigator)
        self.info("Nav2 is ready for use!")

    def _waitForNodeToActivate(self, node_name):
        self.debug(f"Waiting for {node_name} to become active..")
        node_service = f"{node_name}/get_state"
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f"{node_service} service not available, waiting...")

        req = GetState.Request()
        state = "unknown"
        while state != "active":
            self.debug(f"Getting {node_name} state...")
            future = state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f"Result of get_state: {state}")
            time.sleep(2)

    def YawToQuaternion(self, angle_z=0.0):
        quat_tf = quaternion_from_euler(0, 0, angle_z)
        return Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])

    def _amclPoseCallback(self, msg):
        self.debug("Received amcl pose")
        self.initial_pose_received = True
        self.current_pose = msg.pose

    def _feedbackCallback(self, msg):
        self.debug("Received action feedback message")
        self.feedback = msg.feedback

    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def setInitialPose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose = pose
        msg.header.frame_id = self.pose_frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        self.info("Publishing Initial Pose")
        self.initial_pose_pub.publish(msg)

    def info(self, msg):
        self.get_logger().info(msg)

    def warn(self, msg):
        self.get_logger().warn(msg)

    def error(self, msg):
        self.get_logger().error(msg)

    def debug(self, msg):
        self.get_logger().debug(msg)


def parse_positions_file(filename):
    coordinates = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                coordinates.append((x, y))
    return coordinates


def compute_yaw_from_points(x1, y1, x2, y2):
    """Compute the yaw (angle) from point (x1, y1) to (x2, y2)"""
    return math.atan2(y2 - y1, x2 - x1)


def main(args=None):
    rclpy.init(args=args)
    rc = RobotCommander()

    rc.waitUntilNav2Active()

    # Wait until AMCL pose is received
    rc.info("Waiting for AMCL pose...")
    while not rc.initial_pose_received:
        rclpy.spin_once(rc, timeout_sec=0.5)
    rc.info("AMCL pose received!")

    # Wait for dock status
    while rc.is_docked is None:
        rclpy.spin_once(rc, timeout_sec=0.5)

    if rc.is_docked:
        rc.undock()
        time.sleep(2)

    coordinates = parse_positions_file("positions.txt")
    last_yaw = None

    for idx, (x, y) in enumerate(coordinates):
        rc.info(f"Navigating to position {idx+1}/{len(coordinates)}: x={x}, y={y}")

        # Wait for fresh AMCL pose before sending new goal
        rc.initial_pose_received = False
        while not rc.initial_pose_received:
            rclpy.spin_once(rc, timeout_sec=0.5)

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = rc.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y

        # Calculate yaw based on the direction to the next point
        if idx < len(coordinates) - 1:
            next_x, next_y = coordinates[idx + 1]
            yaw = compute_yaw_from_points(x, y, next_x, next_y)
        elif last_yaw is not None:
            yaw = last_yaw
        else:
            # Use AMCL yaw for the first goal if no prior yaw
            q = rc.current_pose.pose.orientation
            _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        goal_pose.pose.orientation = rc.YawToQuaternion(yaw)
        last_yaw = yaw

        rc.info(f"Sending goal with yaw = {round(yaw * 180 / math.pi, 2)}°")

        if not rc.goToPose(goal_pose):
            rc.error("Failed to reach the goal position")
            break

        while not rc.isTaskComplete():
            rc.info("Waiting for the task to complete...")
            time.sleep(1)

        time.sleep(1)

    rc.info("All positions visited!")
    rc.destroyNode()


if __name__ == "__main__":
    main()
