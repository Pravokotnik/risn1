#!/usr/bin/env python3
import rclpy
import heapq
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tasks_msgs.msg import Task  # Custom message type
from yapper import Yapper
from robot_commander import RobotCommander

class HybridController(RobotCommander):
    def __init__(self):
        super().__init__('hybrid_controller')
        
        # Priority queue (max-heap using negative priorities)
        self.task_heap = []
        self.current_task = None
        
        # Single subscriber for all tasks
        self.task_sub = self.create_subscription(
            Task,  # Using our custom message type
            '/tasks',
            self.task_callback,
            10
        )
        
        self.yapper = Yapper()
        self.get_logger().info("Priority-based controller ready")

    def task_callback(self, msg):
        """Handle all incoming tasks with their sender-defined priorities"""
        self.add_task(msg.priority, msg.task_type, msg.target_pose, msg.description)

    def add_task(self, priority, task_type, pose, description=""):
        """Add task to priority queue"""
        entry = (-priority, time.time(), {  # Negative for max-heap
            'type': task_type,
            'pose': pose,
            'description': description
        })
        heapq.heappush(self.task_heap, entry)
        
        # Interrupt current task if new one has higher priority
        if self.current_task and priority > abs(self.current_task[0]):
            self.get_logger().info(f"Interrupting for {description}")
            self.cancel_current_task()

    def get_default_waypoints(self):
        """Default waypoint route"""
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

    def cancel_current_task(self):
        """Cancel current task and requeue it"""
        if self.current_task:
            heapq.heappush(self.task_heap, self.current_task)
            self.cancelTask()
            self.current_task = None

    # def determine_color(self, color_msg):
    #     """Convert color message to name"""
    #     if color_msg.r > 0.5: return "red"
    #     if color_msg.g > 0.5: return "green"
    #     if color_msg.b > 0.5: return "blue"
    #     if sum([color_msg.r, color_msg.g, color_msg.b]) < 0.3: return "black"
    #     return "unknown"

    def initialize_robot(self):
        """Initialize and undock"""
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
                self.process_tasks()
        finally:
            self.destroyNode()
            rclpy.shutdown()

    def process_tasks(self):
        """Process highest priority task"""
        if not self.current_task and self.task_heap:
            self.current_task = heapq.heappop(self.task_heap)
            self.execute_task(self.current_task[2])

    def execute_task(self, task):
        """Execute navigation task"""
        try:
            if self.goToPose(task['pose']):
                self.handle_task_behavior(task)
                self.wait_for_completion(task['description'])
        except Exception as e:
            self.get_logger().error(f"Task failed: {str(e)}")
        finally:
            self.current_task = None

    def handle_task_behavior(self, task):
        """Task-specific behaviors"""
        if task['priority'] == PRIORITY_RING:
            self.yapper.yap(f"Found {task.get('color', 'unknown')} ring!")
        elif task['priority'] == PRIORITY_FACE:
            self.yapper.yap("Hello there!")

    def wait_for_completion(self, description):
        """Wait for task completion with interrupt check"""
        while not self.isTaskComplete():
            self.get_logger().info(
                f"{description}... "
                f"Position: {self.current_pose.pose.position.x:.2f}, "
                f"{self.current_pose.pose.position.y:.2f}",
                throttle_duration_sec=2.0
            )
            time.sleep(0.1)

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