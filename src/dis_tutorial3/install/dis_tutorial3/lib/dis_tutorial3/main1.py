#!/usr/bin/env python3
import rclpy
import heapq
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from dis_tutorial3.msg import Task  # Custom message type
from yapper import Yapper
from robot_commander import RobotCommander

class HybridController(RobotCommander):
    def __init__(self):
        super().__init__('hybrid_controller')
        
        # Priority queue (max-heap using negative priorities)
        self.task_heap = []
        self.current_task = None
        
        # Subscriber for all tasks
        self.task_sub = self.create_subscription(
            Task,
            '/tasks',
            self.task_callback,
            10
        )
        
        # Publisher for RViz markers
        self.marker_pub = self.create_publisher(MarkerArray, '/task_markers', 10)
        
        self.yapper = Yapper()
        self.get_logger().info("Priority-based controller ready")

    def task_callback(self, msg):
        """Handle all incoming tasks with their sender-defined priorities"""
        self.add_task(msg.priority, msg.task_type, msg.target_pose, msg.description)

    def add_task(self, priority, task_type, pose, description=""):
        """Add task to priority queue, then refresh RViz markers"""
        entry = (-priority, time.time(), {
            'type': task_type,
            'pose': pose,
            'description': description
        })
        heapq.heappush(self.task_heap, entry)
        
        # If a higher-priority task arrives, interrupt
        if self.current_task and -priority > self.current_task[0]:
            self.get_logger().info(f"Interrupting for {description}")
            self.cancel_current_task()

        # Update markers for the entire queue
        self.publish_task_markers()

    def cancel_current_task(self):
        """Cancel current task, requeue it, and refresh markers"""
        if self.current_task:
            heapq.heappush(self.task_heap, self.current_task)
            self.cancelTask()
            self.current_task = None
            self.publish_task_markers()

    def publish_task_markers(self):
        """Publish a MarkerArray, one arrow per queued task"""
        marray = MarkerArray()
        for idx, entry in enumerate(self.task_heap):
            priority, _, task = entry
            # real priority = -priority
            prio = -priority
            pose = task['pose']
            
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'tasks'
            m.id = idx
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose = pose.pose
            # Scale: length=0.5m, width=0.1m
            m.scale.x = 0.5
            m.scale.y = 0.1
            m.scale.z = 0.1
            
            # Color map: high priority = red, medium = yellow, low = green
            color = ColorRGBA()
            if prio >= 4:
                color.r, color.g, color.b, color.a = 1.0, 0.0, 0.0, 0.8
            elif prio >= 2:
                color.r, color.g, color.b, color.a = 1.0, 1.0, 0.0, 0.8
            else:
                color.r, color.g, color.b, color.a = 0.0, 1.0, 0.0, 0.8
            m.color = color
            
            marray.markers.append(m)
        
        self.marker_pub.publish(marray)

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
            # … etc …
        ]

    def initialize_robot(self):
        """Initialize and undock, then queue default waypoints"""
        self.waitUntilNav2Active()
        while self.is_docked is None:
            rclpy.spin_once(self, timeout_sec=0.5)
        if self.is_docked:
            self.undock()
            while self.is_docked:
                rclpy.spin_once(self, timeout_sec=0.5)
        
        for waypoint in self.get_default_waypoints():
            self.add_task(0, "waypoint", waypoint, "Default waypoint")

    def run(self):
        """Main execution loop"""
        self.initialize_robot()
        self.get_logger().info("Starting main loop")
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)
                self.process_tasks()
        finally:
            self.destroy_node()
            rclpy.shutdown()

    def process_tasks(self):
        """Pop and execute the highest-priority task"""
        if not self.current_task and self.task_heap:
            self.current_task = heapq.heappop(self.task_heap)
            # Refresh markers so the just-started task disappears
            self.publish_task_markers()
            self.execute_task(self.current_task[2])

    def execute_task(self, task):
        """Navigate to the pose and handle post-arrival behavior"""
        try:
            if self.goToPose(task['pose']):
                self.handle_task_behavior(task)
                self.wait_for_completion(task['description'])
        except Exception as e:
            self.get_logger().error(f"Task failed: {e}")
        finally:
            self.current_task = None

    def handle_task_behavior(self, task):
        """Optional post-arrival actions (e.g. speech)"""
        # if task['type'] == "face":
        #     self.yapper.yap("Hello there!")

    def wait_for_completion(self, description):
        """Wait until the navigation goal is reached, logging periodically"""
        while not self.isTaskComplete():
            self.get_logger().info(
                f"{description}… "
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
