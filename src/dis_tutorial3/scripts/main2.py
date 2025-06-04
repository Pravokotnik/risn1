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
        self.marker_pub = self.create_publisher(MarkerArray, 'waypoints', 10)
        
        
        
        self._next_marker_id = 500
        
        self.canceledTask = False
        
        self.yapper = Yapper()
        self.get_logger().info("Priority-based controller ready")
        
        self.seen_ids = {}

    def task_callback(self, msg):
        self.add_task(msg.priority, msg.task_type, msg.target_pose, msg.description, task_id=msg.id)


    def add_task(self, priority, task_type, pose, description="", task_id=None):
        # Search for existing task with same priority, type, and id
        for idx, entry in enumerate(self.task_heap):
            if task_id is None:
                continue
            # Check if task id and task type already in seen_ids dict
            if not (task_id, task_type) in self.seen_ids:
                continue
            prio, _, task = entry
            if -prio == priority and task['type'] == task_type and task.get('id') == task_id:
                # Update this existing task
                new_entry = (-priority, time.time(), {
                    'type': task_type,
                    'pose': pose,
                    'description': description,
                    'id': task_id,
                    'priority': priority
                })
                self.task_heap[idx] = new_entry
                heapq.heapify(self.task_heap)  # Rebuild heap after modification
                self.get_logger().info(f"Updated existing task {task_type} with id {task_id} and priority {priority}")
                
                # If updated task is currently running, interrupt it
                if self.current_task and self.current_task[2]['id'] == task_id and self.current_task[2]['type'] == task_type:
                    self.get_logger().info(f"Interrupting current task {task_type} with id {task_id} due to update")
                    self.cancel_current_task()
                break
        else:
            if not (task_id, task_type) in self.seen_ids:
                # No existing task with same priority and id found; add new
                entry = (-priority, time.time(), {
                    'type': task_type,
                    'pose': pose,
                    'description': description,
                    'id': task_id,
                    'priority': priority
                })
                heapq.heappush(self.task_heap, entry)
                self.get_logger().info(f"Added new task {task_type} with id {task_id} and priority {priority}")

        if task_id is not None:
            self.seen_ids[task_id, task_type] = True
        r = 0.0
        g = 0.0
        b = 0.0
        
        if task_type == "face":
            r, g, b = 1.0, 0.5, 0.5
        elif task_type == "waypoint":
            r, g, b = 0.2, 1.0, 0.2
        elif task_type == "speech":
            r, g, b = 0.2, 0.2, 1.0
        elif task_type == "emergency":
            r, g, b = 1.0, 0.0, 0.0
        else:
            r, g, b = 0.5, 0.5, 0.5
        
        self.publish_to_map(pose.pose.position, r=r, g=g, b=b, a=0.9)
        
        # If a higher-priority task arrives, interrupt
        if self.current_task and -priority < self.current_task[0]:
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
            self.canceledTask = True
            self.get_logger().warn("Current task cancelled and requeued")
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
            create_pose(0.87, 5.64, -0.00),
            create_pose(-3.41, 6.00, -0.00),
            create_pose(-2.98, 1.77, -0.00),
        ]

    def initialize_robot(self):
        """Initialize and undock, then queue default waypoints"""
        self.waitUntilNav2Active()
        # while self.is_docked is None:
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
            completed = False
            if task['type'] == "ring":
                completed = self.goToPoseProximity(task['pose'], 0.3)
            else:
                completed = self.goToPose(task['pose'])
            if completed:
                self.handle_task_behavior(task)
                self.wait_for_completion(task['description'])
                # Spin 360 degrees if it's a waypoint task
                if not self.canceledTask:
                    if task['type'] == "waypoint":
                        self.spin(360.0)
                        self.wait_for_completion("Spinning 360 degrees")
                    # Say "Hello!" if it's a face task
                    elif task['type'] == "face":
                        self.get_logger().error("YAPPING!")
                        self.yapper.yap("Hello there!")
                    # Wait 1s if emergency task
                    elif task['type'] == "emergency":
                        time.sleep(2.0)
                self.canceledTask = False
            else:
                self.get_logger().error(f"Failed to reach {task['description']}")
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
        m.scale.x = m.scale.y = m.scale.z = 0.2
        m.color = ColorRGBA(r=r, g=g, b=b, a=a)
        arr = MarkerArray()
        arr.markers.append(m)
        self.marker_pub.publish(arr)
        
        self.get_logger().info(f"Published marker at {map_point.x}, {map_point.y}, {map_point.z}")

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
