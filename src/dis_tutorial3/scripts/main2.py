#!/usr/bin/env python3
import math

import cv2
import rclpy
import heapq
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from dis_tutorial3.msg import Task  # Custom message type
from yapper import Yapper
from robot_commander import RobotCommander
import re
from dis_tutorial3.msg import FaceMsg  # Custom message type for face detection. ONly used for gender detection

import speech_recognition as sr

from bird_classifier import predict_bird_name

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
        self.gender_sub = self.create_subscription(
            FaceMsg,
            '/face_coordinates',
            self.gender_callback,
            10
        )
        self.arm_camera_sub = self.create_subscription(
            Image,
            '/top_camera/rgb/preview/image_raw',
            self.arm_image_callback,
            qos_profile=QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                depth=1
            )
        )
        self.latest_arm_image = None
        
        # Publisher for RViz markers
        self.marker_pub = self.create_publisher(MarkerArray, 'waypoints', 10)
        self.arm_pub = self.create_publisher(String, '/arm_command', 10)
        
        arm_msg = String()
        COMMAND = "manual:[0.0,0.0,0.0,1.57]"
        arm_msg.data = COMMAND
        self.arm_pub.publish(arm_msg)
        self.get_logger().info(f'Published "{COMMAND}" to /arm_command')
        
        #Speech to text
        self.stt_recognizer = sr.Recognizer()
        
        self.bird_names = [
            "laysan albatross",
            "yellow headed blackbird",
            "indigo bunting",
            "pelagic cormorant",
            "american crow",
            "yellow billed cuckoo",
            "purple finch",
            "vermilion flycatcher",
            "european goldfinch",
            "eared grebe",
            "california gull",
            "ruby throated hummingbird",
            "blue jay",
            "pied kingfisher",
            "baltimore oriole",
            "white pelican",
            "horned puffin",
            "white necked raven",
            "great grey shrike",
            "house sparrow",
            "cape glossy starling",
            "tree swallow",
            "common tern",
            "red headed woodpecker"
        ]
        
        
        
        
        self._next_marker_id = 500
        
        self.canceledTask = False
        self.last_detected_gender = None
        
        self.yapper = Yapper()
        self.get_logger().info("Priority-based controller ready")
        
        self.seen_ids = {}

    def task_callback(self, msg):
        self.add_task(msg.priority, msg.task_type, msg.target_pose, msg.description, task_id=msg.id)
    
    def gender_callback(self, msg):
        gender = msg.gender.lower()
        self.last_detected_gender = gender
    
    def arm_image_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        try:
            # Convert ROS Image to OpenCV image (BGR8)
            self.latest_arm_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return


    def extract_gender_from_description(self, description):
        """Extract gender from task description"""
        if not description:
            return "unknown"
        
        # Look for "Gender: male/female/unknown" pattern in description
        gender_match = re.search(r'Gender:\s*(\w+)', description, re.IGNORECASE)
        if gender_match:
            gender = gender_match.group(1).lower()
            if gender in ['male', 'female', 'unknown']:
                return gender
        
        return "unknown"
    
    def get_greeting_for_gender(self, gender):
        """Get appropriate greeting based on gender"""
        if gender == "male":
            return "Hey man!"
        elif gender == "female":
            return "Hey woman!"
        else:
            return "Hey!"


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
            create_pose(2.07, 1.35, -0.00),
            create_pose(-0.60, 1.96, 0.00),
            create_pose(-0.58, 4.17, -0.00),
            create_pose(1.07, 4.34, 0.00),
            create_pose(0.65, 5.87, -0.00),
            create_pose(-2.85, 6.17, -0.00),
            create_pose(-3.07, 2.77, -0.00),
            create_pose(-2.37, 1.39, 0.00),
            create_pose(0.0, 0.0, 0.00),
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
            self.add_task(1, "waypoint", waypoint, "Default waypoint")

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
                completed = self.goToPoseProximity(task['pose'], 0.6)
            else:
                completed = self.goToPose(task['pose'])
            if completed:
                self.handle_task_behavior(task)
                self.wait_for_completion(task['description'])
                # Spin 360 degrees if it's a waypoint task
                if not self.canceledTask:
                    self.execute_specialized_behavior(task)
                self.canceledTask = False
            else:
                self.get_logger().error(f"Failed to reach {task['description']}")
        except Exception as e:
            self.get_logger().error(f"Task failed: {e}")
        finally:
            self.current_task = None
    
    def execute_specialized_behavior(self, task):
        if task['type'] == "waypoint":
            self.spin(360.0)
            self.wait_for_completion("Spinning 360 degrees")
        # Say "Hello!" if it's a face task
        elif task['type'] == "face":
            self.execute_face_behavior(task)
        # Wait 1s if emergency task
        elif task['type'] == "emergency":
            time.sleep(2.0)
        elif task['type'] == "ring":
            self.execute_ring_behavior(task)
    
    def execute_face_behavior(self, task):
        time.sleep(1.0)  # Wait for guaranteed gender detection
        gender = self.last_detected_gender
        greeting = self.get_greeting_for_gender(gender)
        self.get_logger().info(f"Detected gender: {gender}, saying: {greeting}")
        self.yapper.yap(greeting + ", which is your favorite bird?")

        selected_birds = []
        while True:
            # Get speech input from the user
            text = self.get_sst_text()
            if text is None or text.strip() == "":
                self.yapper.yap("I didn't understand that. Please try again.")
                continue
            
            # Check if the response contains a bird name
            if self.check_for_bird_in_text(text):
                mentioned_bird = None
                for bird in self.bird_names:
                    if bird in text.lower():
                        mentioned_bird = bird
                        break
                
                # If they are male and already mentioned the bird before, the answer is now locked in
                if mentioned_bird in selected_birds:
                    selected_birds.append(mentioned_bird)
                    break
                
                selected_birds.append(mentioned_bird)
                # Females only answer once
                if gender == "female":
                    break
                
                # Male only interaction
                self.yapper.yap(f"You like {selected_birds[-1]}. Are you sure?")
                            
            elif "yes" in text.lower() or "sure" in text.lower():
                if len(selected_birds) == 0:
                    self.yapper.yap("You didn't mention any bird. Please try again.")
                else:
                    break
            else:
                self.yapper.yap("I didn't understand that. Please mention a bird name or say 'yes' to confirm.")
        
        self.yapper.yap(f"Great! You like {selected_birds[-1]}. I will remember that.")
        self.yapper.yap("Kill yourself.")
        time.sleep(5.0)

    
    def get_sst_text(self):
        # 2. Grab audio from the default microphone
        with sr.Microphone() as source:
            self.get_logger().info("Please speak something...")
            audio_data = self.stt_recognizer.listen(source)
            self.get_logger().info("Audio captured, processing...")

        # 3. Recognize speech using Sphinx
        try:
            text = self.stt_recognizer.recognize_google(audio_data)
            self.get_logger().info(f"Recognized text: {text}")
            return text
        except sr.UnknownValueError:
            self.get_logger().error("Speech Recognition could not understand audio")
        except Exception as e:
            self.get_logger().error(f"Could not request results from Speech Recognition service; {e}")
        return None
    
    def check_for_bird_in_text(self, text):
        pass
    
    def execute_ring_behavior(self, task):
        # First get the ring position from the description
        description = task['description']
        description_split = description.split('|')
        if len(description_split) < 2:
            self.get_logger().error("Invalid ring description format")
            return
        # Get x and y coordinates from last part of the description
        coords = description_split[-1].strip()
        coords_split = coords.split(',')
        if len(coords_split) != 2:
            self.get_logger().error("Invalid ring coordinates format")
            return
        try:
            x = float(coords_split[0].strip())
            y = float(coords_split[1].strip())
        except ValueError:
            self.get_logger().error("Invalid ring coordinates values")
            return
        
        # Rotate to face the ring
        self.get_logger().info(f"Rotating to face ring at ({x}, {y})")
        
        self.rotate_towards_point(x, y)
        time.sleep(1.0)
        
        image = self.latest_arm_image
        if image is None:
            self.get_logger().error("No image received from arm camera")
            return
        
        # Put image through AI model to detect bird species
        bird_name = predict_bird_name(image)
        self.get_logger().info(f"Detected bird species: {bird_name}")
        
        cv2.imshow(f"Bird: {bird_name}", image)
        cv2.waitKey(0)
        
        
    def rotate_towards_point(self, target_x, target_y):
        """Rotate to face a specific position"""
        current_x = self.current_pose.pose.position.x
        current_y = self.current_pose.pose.position.y
        angle_to_target = self.get_angle_to_target(current_x, current_y, target_x, target_y)
        
        # Convert angle to quaternion
        z = math.sin(angle_to_target/2)
        w = math.cos(angle_to_target/2)
        
        # Create a new pose with the target orientation
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'map'
        target_pose.pose.position.x = current_x
        target_pose.pose.position.y = current_y
        target_pose.pose.orientation.z = z
        target_pose.pose.orientation.w = w
        self.get_logger().info(f"Rotating to face position ({target_x}, {target_y}) with angle {angle_to_target:.2f} radians")
        self.goToPose(target_pose)
        self.wait_for_completion("Rotating to face position")
        time.sleep(10.0)  # Give some time to stabilize after rotation
        
    
    def get_angle_to_target(self, current_x, current_y, target_x, target_y):
        """Calculate the angle to face a target position"""
        delta_x = target_x - current_x
        delta_y = target_y - current_y
        angle = math.atan2(delta_y, delta_x)
        return angle

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
