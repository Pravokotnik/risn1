#!/usr/bin/env python3
import math
import os
import re
import time
import heapq
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String
from sensor_msgs.msg import Image as ImageMsg

from dis_tutorial3.msg import Task, FaceMsg
from yapper import Yapper
from robot_commander import RobotCommander

import speech_recognition as sr
from ament_index_python import get_package_share_directory
from cv_bridge import CvBridge

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from skimage.morphology import skeletonize



# ────────────────────────────────────────────────────────────────────────────────
# ─── 1. Load bird‐classifier model and mapping (runs at import time) ────────────

# find the package root (one level up from scripts/)
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

# locate the installed bird_checkpoints folder under share/dis_tutorial3
share_dir    = get_package_share_directory('dis_tutorial3')
ckpt_folder  = Path(share_dir) / 'bird_checkpoints_updated'
CHECKPOINT_PATH = str(ckpt_folder / 'best_model_updated.pth')
MAPPING_PATH    = str(ckpt_folder / 'idx_to_class_updated.pth')

INPUT_SIZE = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_mapping(checkpoint_path: str, mapping_path: str, device: torch.device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at:\n  {checkpoint_path}")
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(f"Mapping file not found at:\n  {mapping_path}")

    idx_to_class = torch.load(mapping_path, map_location=device)
    num_classes = len(idx_to_class)

    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, idx_to_class

_MODEL, _IDX_TO_CLASS = load_model_and_mapping(CHECKPOINT_PATH, MAPPING_PATH, DEVICE)


# ────────────────────────────────────────────────────────────────────────────────
# ─── 2. Cropping + Preprocessing Helpers ────────────────────────────────────────

def crop_bird_region(img_bgr: np.ndarray) -> Image.Image | None:
    height, width = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 120,  60])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return None

    (cx, cy), r = cv2.minEnclosingCircle(largest)
    cx, cy, r = float(cx), float(cy), float(r)
    if r < 10 or r > min(width, height) / 2:
        return None

    left   = int(max(0, cx - 0.8 * r))
    right  = int(min(width, cx + 0.8 * r))
    top    = int(max(0, cy - 1.5 * r))
    bottom = int(min(height, cy - 0.2 * r))

    if left >= right or top >= bottom:
        return None

    rgb_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return rgb_pil.crop((left, top, right, bottom))


def top_center_crop(img_bgr: np.ndarray) -> Image.Image:
    height, width = img_bgr.shape[:2]
    crop_size = INPUT_SIZE
    cx = width // 2
    cy = height // 4

    left   = max(0, cx - crop_size // 2)
    right  = min(width, cx + crop_size // 2)
    top    = max(0, cy - crop_size // 2)
    bottom = min(height, top + crop_size)

    if right - left < crop_size:
        left = max(0, right - crop_size)
    if bottom - top < crop_size:
        top = max(0, bottom - crop_size)

    rgb_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return rgb_pil.crop((left, top, right, bottom))


def center_crop_full(img_bgr: np.ndarray) -> Image.Image:
    height, width = img_bgr.shape[:2]
    crop_size = INPUT_SIZE

    left   = max(0, width // 2 - crop_size // 2)
    top    = max(0, height // 2 - crop_size // 2)
    right  = min(width, left + crop_size)
    bottom = min(height, top + crop_size)

    if right - left < crop_size:
        left = max(0, right - crop_size)
    if bottom - top < crop_size:
        top = max(0, bottom - crop_size)

    rgb_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return rgb_pil.crop((left, top, right, bottom))


_preproc = transforms.Compose([
    transforms.Resize(int(INPUT_SIZE * 1.14)),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict_topk_on_crop(model: torch.nn.Module,
                         idx_to_class: dict[int, str],
                         pil_crop: Image.Image,
                         topk: int,
                         device: torch.device) -> list[tuple[str, float]]:
    tensor = _preproc(pil_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)

    top_probs, top_idxs = probs.topk(topk, dim=1)
    top_probs = top_probs.cpu().squeeze(0)
    top_idxs  = top_idxs.cpu().squeeze(0)

    results = []
    for idx, p in zip(top_idxs, top_probs):
        species = idx_to_class[int(idx)]
        results.append((species, float(p.item())))
    return results


def predict_topk(img_bgr: np.ndarray, topk: int = 3) -> list[tuple[str, float]]:
    crops: list[Image.Image] = []
    hsv_crop = crop_bird_region(img_bgr)
    if hsv_crop is not None:
        crops.append(hsv_crop)

    crops.append(top_center_crop(img_bgr))
    crops.append(center_crop_full(img_bgr))

    best_conf = -1.0
    best_crop: Image.Image | None = None
    for crop in crops:
        top1, prob = predict_topk_on_crop(_MODEL, _IDX_TO_CLASS, crop, 1, DEVICE)[0]
        if prob > best_conf:
            best_conf = prob
            best_crop = crop

    if best_crop is None:
        raise RuntimeError("Failed to generate any valid crop for bird classification.")
    return predict_topk_on_crop(_MODEL, _IDX_TO_CLASS, best_crop, topk, DEVICE)


def predict_bird_name(img_bgr: np.ndarray) -> str:
    top1_species, _ = predict_topk(img_bgr, topk=1)[0]
    return top1_species


# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# ─── 3. Your HybridController Node ───────────────────────────────────────────────

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
            ImageMsg,
            '/top_camera/rgb/preview/image_raw',
            self.arm_image_callback,
            qos_profile=QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                depth=1
            )
        )
        # Subscribe to depth image
        self.depth_sub = self.create_subscription(
            ImageMsg,
            '/top_camera/rgb/preview/depth',
            self.depth_image_callback,
            qos_profile=QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                depth=1
            )
        )
        self.latest_arm_image = None
        self.latest_depth_image = None
        
        if not hasattr(self, 'vel_publisher'):
            self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        
        # Publisher for RViz markers
        self.marker_pub = self.create_publisher(MarkerArray, 'waypoints', 10)
        self.arm_pub = self.create_publisher(String, '/arm_command', 10)
        
        arm_msg = String()
        COMMAND = "manual:[0.0,0.0,0.0,1.57]"
        arm_msg.data = COMMAND
        self.arm_pub.publish(arm_msg)
        self.get_logger().info(f'Published "{COMMAND}" to /arm_command')
        
        self.bridge = CvBridge()
        
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
        
        self.photographed_birds = {}
        self.bird_catalogue = {}
        
        
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
    
    def depth_image_callback(self, msg):
        """Callback for depth image"""
        try:
            # Convert ROS Image to OpenCV image (32FC1 depth)
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Could not convert depth image: {e}')


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
            create_pose(-3.06, 7.19, -0.00),
            create_pose(-3.65, 1.62, -0.00),
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
            self.add_task(20, "waypoint", waypoint, "Default waypoint")
            # pass
            
        # Add lowest priority bridge task to go to coordinates -0.01, -0.83, 0.36
        final_waypoint = PoseStamped()
        final_waypoint.header.frame_id = 'map'
        final_waypoint.pose.position.x = -0.01
        final_waypoint.pose.position.y = -0.83
        final_waypoint.pose.position.z = 0.36
        final_waypoint.pose.orientation = self.YawToQuaternion(math.pi*2*3/4)  # Face east
        self.add_task(1, "bridge", final_waypoint, "Final waypoint to face east", task_id=9999)

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
        elif task['type'] == "bridge":
            self.execute_bridge_behavior(task)
    
    def execute_bridge_behavior(self, task):
        self.get_logger().info("Creating bird catalogue.")
        self.export_catalogue_to_pdf(self.bird_catalogue, 'bird_catalogue.pdf')

        # 1) Position arm camera slightly forward + down for RGB view
        arm_msg = String()
        # (tune these four numbers until the camera sees enough of the bridge path above the robot)
        arm_msg.data = "manual:[0.0,0.37,1.2,1.28]"
        self.arm_pub.publish(arm_msg)
        self.get_logger().info("Positioned arm camera to view bridge")
        time.sleep(6.0)

        # ---- Reset PID state before crossing ----
        self.prev_offset = 0.0
        self.integral    = 0.0

        bridge_complete = False
        while not bridge_complete and not self.canceledTask:
            rclpy.spin_once(self, timeout_sec=0.01)
            # ─── 2) WAIT FOR A VALID RGB IMAGE ─────────────────────────────────
            if self.latest_arm_image is None:
                self.get_logger().warn("No RGB image yet → stopping robot")
                self.send_velocity_cmd(0.0, 0.0)
                time.sleep(0.2)
                continue

            # ─── 3) BUILD A BINARY “SAFE” MASK FROM RGB ────────────────────────
            # (blue or green = danger; everything else in top 70% = safe)
            binary_img = self.process_rgb_image(self.latest_arm_image)
            
            # Display the binary image for debugging
            cv2.imshow("Binary Safe Mask", binary_img)
            cv2.waitKey(1)  # Wait for a short time to allow the image to be displayed

            # ─── 4) IF THE “SAFE” MASK IS TOO SMALL → STOP & RETRY ─────────────
            if cv2.countNonZero(binary_img) < 200:
                self.get_logger().warn("Safe‐mask too sparse → stopping robot")
                self.send_velocity_cmd(0.0, 0.0)
                time.sleep(0.2)
                continue

            # ─── 5) FIND CENTERLINE VIA CONTOURS (REPLACES SKELETONIZATION) ──
            center_x, img_w = self.find_centerline_via_contours(binary_img)
            if center_x is None:
                self.get_logger().warn("No bridge contour found → stopping robot")
                self.send_velocity_cmd(0.0, 0.0)
                time.sleep(0.2)
                continue

            img_center = img_w / 2.0
            offset = center_x - img_center
            # ─── END OF CONTOUR SNIPPET ───────────────────────────────────────

            # ─── 6) APPLY PID USING THAT OFFSET ───────────────────────────────
            linear_vel, angular_vel = self.bridge_pid_control(offset)
            self.send_velocity_cmd(linear_vel, angular_vel)

            # ─── 7) CHECK FOR COMPLETION (e.g. open area ahead or distance) ──
            bridge_complete = self.check_bridge_completion()
            time.sleep(0.05)

        # ─── 8) STOP VEHICLE ONCE DONE (OR CANCELED) ─────────────────────────
        self.send_velocity_cmd(0.0, 0.0)
        self.get_logger().info("Bridge crossing complete")
        
            
    def export_catalogue_to_pdf(self, catalogue: dict[str, any], output_pdf_path: str) -> None:
        """
        Given a dict mapping bird_name (str) → cv2_image (np.ndarray, BGR),
        export one page per entry into a PDF, with the image and the name below.
        """
        # PdfPages will collect multiple pages into a single PDF.
        with PdfPages(output_pdf_path) as pdf:
            for bird_name, img_bgr in catalogue.items():
                # 1. Convert BGR → RGB for Matplotlib
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # 2. Create a new figure and show the image
                plt.figure(figsize=(6, 6))
                plt.imshow(img_rgb)
                plt.axis("off")

                # 3. Put the bird's name below (as title or text)
                # Option A: use title (centered above/below the image)
                plt.title(bird_name.replace("_", " ").title(), fontsize=14, pad=10)

                # ↓ Option B: if you want the text _below_ the image instead:
                # plt.text(
                #     0.5, -0.05, 
                #     bird_name.replace("_", " ").title(), 
                #     ha="center", va="top",
                #     transform=plt.gca().transAxes,
                #     fontsize=12
                # )

                # 4. Save this figure as one page in the PDF
                pdf.savefig(bbox_inches="tight")
                plt.close()

        print(f"Saved catalogue PDF to: {output_pdf_path}")

    def process_depth_image(self, depth_img):
        depth_copy = np.nan_to_num(depth_img, nan=10.0)
        h, w = depth_copy.shape

        # Ignore bottom 25% of rows (where robot appears)
        crop_top = int(0.70 * h)
        roi = depth_copy[0:crop_top, :]

        # Take a front‐facing sub‐ROI (middle third horizontally, lower half of the kept rows)
        front_start = int(0.5 * crop_top)
        front_roi = roi[front_start:crop_top, int(w/3):int(2*w/3)]

        # Discard too‐near (<0.2 m) or too‐far (>2.0 m)
        valid = front_roi[(front_roi > 0.2) & (front_roi < 2.0)]
        if len(valid) == 0:
            return np.zeros_like(depth_copy, dtype=np.uint8)

        bridge_depth = np.median(valid)
        self.get_logger().info(f"Bridge depth ≈ {bridge_depth:.2f} m")

        tol = 0.10  # ±30 cm tolerance
        mask_top75 = np.abs(roi - bridge_depth) < tol

        binary = np.zeros_like(depth_copy, dtype=np.uint8)
        binary[0:crop_top, :][mask_top75] = 255

        kernel = np.ones((7,7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

        return binary

    def process_rgb_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Convert a BGR image into a binary mask of “safe” pixels (path) 
        by ignoring bottom 30% and marking blue (water) or green (grass) as danger.
        Returns a single‐channel uint8 image where 255=“safe” and 0=“danger/ignored”.
        """
        # 1) Copy & compute dimensions
        img_copy = img_bgr.copy()
        height, width = img_copy.shape[:2]

        # 2) Ignore bottom 30% (where the robot’s body might appear)
        crop_top = int(0.70 * height)          # keep rows 0..(0.70*h - 1)
        roi = img_copy[0:crop_top, :]          # drop bottom 30%

        # 3) Convert that ROI to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 4) Build blue mask (water) in ROI
        #    – you can tweak HSV thresholds if needed
        lower_blue = np.array([110, 30, 100])
        upper_blue = np.array([130, 255, 255])

        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # 5) Build green mask (grass) in ROI
        lower_green = np.array([20, 80,  76])
        upper_green = np.array([40, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # 6) Combine “danger” = blue OR green
        danger_mask_roi = cv2.bitwise_or(mask_blue, mask_green)

        # 7) “Safe” mask is the inverse of danger, within the same ROI
        safe_mask_roi = cv2.bitwise_not(danger_mask_roi)

        # 8) Create a full‐image binary, but only fill the top 70% from safe_mask_roi
        binary = np.zeros((height, width), dtype=np.uint8)
        #  safe_mask_roi is single‐channel, size = (crop_top, width)
        binary[0:crop_top, :][ safe_mask_roi > 0 ] = 255

        # 9) Clean small holes/noise
        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

        return binary
    
    
    def find_centerline_via_contours(self, binary_img):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        main = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main)
        center_x = x + w/2.0
        return center_x, binary_img.shape[1]

    def check_bridge_completion(self) -> bool:
        """
        Return True as soon as we see the exact color 0xFF655B (BGR = [91,101,255])
        anywhere in the latest RGB frame.
        """
        if self.latest_arm_image is not None:
            # Define the exact BGR color to match
            target_bgr = np.array([91, 101, 255], dtype=np.uint8)

            # Create a mask where pixels exactly equal [91,101,255]
            mask_target = cv2.inRange(self.latest_arm_image, target_bgr, target_bgr)

            if cv2.countNonZero(mask_target) > 0:
                self.get_logger().info("Detected ff655b → stopping bridge crossing")
                return True

        return False



    def process_bridge_image(self, image):
        """Convert RGB/depth image to binary bridge mask"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to separate bridge from surroundings
        # You may need to adjust these thresholds based on your environment
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        # Optional: Apply morphological operations to clean the binary image
        kernel = np.ones((5,5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def find_centerline_via_contours(self, binary_img):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        main = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main)
        center_x = x + w / 2.0
        return center_x, binary_img.shape[1]  # return (center_x, image_width)

    def find_bridge_centerline(self, binary_img):
        """Find the centerline of the bridge using skeletonization"""
        
        # Normalize binary image for skeletonization (0 and 1 values)
        normalized = binary_img.copy() / 255
        
        # Apply skeletonization
        skeleton = skeletonize(normalized)
        
        # Convert back to uint8 for OpenCV processing
        skeleton_img = (skeleton * 255).astype(np.uint8)
        
        # If no skeleton found, return None
        if np.sum(skeleton_img) == 0:
            return None
            
        return skeleton_img

    def calculate_centerline_offset(self, centerline_img):
        """Calculate robot's offset from the centerline"""
        # Find centerline points
        height, width = centerline_img.shape
        
        # We're interested in the bottom half of the image (closest to robot)
        lower_half = centerline_img[height//2:, :]
        
        # Find all non-zero points (centerline points)
        points = np.where(lower_half > 0)
        
        if len(points[0]) == 0:
            return 0  # No centerline points found
        
        # Calculate average x position of centerline
        avg_x = np.mean(points[1])
        
        # Calculate offset from center of image
        center_x = width / 2
        offset = avg_x - center_x
        
        return offset

    def bridge_pid_control(self, offset, kp=0.005, kd=0.0005, ki=0.0001):
        """Apply PID control with dead zone for stability"""
        # Static variables for PID
        if not hasattr(self, 'prev_offset'):
            self.prev_offset = 0
        if not hasattr(self, 'integral'):
            self.integral = 0
        
        # Create a dead zone - ignore very small offsets
        dead_zone = 10  # pixels
        if abs(offset) < dead_zone:
            offset = 0
            self.integral = 0  # Reset integral when in dead zone
        
        # PID calculations with much lower gains
        proportional = offset
        derivative = offset - self.prev_offset
        self.integral += offset
        
        # Prevent integral windup
        max_integral = 50  # Reduced from 100
        self.integral = max(min(self.integral, max_integral), -max_integral)
        
        # Try positive sign if negative doesn't work
        angular_velocity = (kp * proportional + kd * derivative + ki * self.integral)
        angular_velocity = -angular_velocity  # Reverse direction for right turn
        
        # Very limited maximum angular velocity
        max_angular = 0.2  # Low maximum angular velocity
        angular_velocity = max(min(angular_velocity, max_angular), -max_angular)
        
        self.get_logger().info(f"Offset: {offset:.2f}, Angular vel: {angular_velocity:.4f}")
        
        self.prev_offset = offset
        
        # Very slow forward speed
        linear_velocity = 0.05
        
        return linear_velocity, angular_velocity

    def send_velocity_cmd(self, linear_vel, angular_vel):
        """Send velocity commands to the robot"""
        # Use your existing command methods or implement a new one
        self.setSpeed(linear_vel, angular_vel)
    
    def setSpeed(self, linear_vel, angular_vel):
        """Publish velocity commands to the robot's cmd_vel topic"""
        
        # Create Twist message
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        
        # Publish the message
        self.vel_publisher.publish(twist_msg)
        self.get_logger().debug(f'Published velocity command: linear={linear_vel}, angular={angular_vel}')
    
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
            mentioned_bird = self.check_for_bird_in_text(text)
            if mentioned_bird is not None:
                
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
        
        selected_bird = selected_birds[-1]
        self.get_logger().info(f"Selected bird: {selected_bird}")
        if selected_bird not in self.photographed_birds:
            self.yapper.yap(f"I'm sorry, I don't have a photo of {selected_bird}.")
        else:
            message = f"There is a {selected_bird} sitting on a {self.photographed_birds[selected_bird]['color']} ring at around {self.photographed_birds[selected_bird]['pose'].pose.position.x:.2f}, {self.photographed_birds[selected_bird]['pose'].pose.position.y:.2f}."
            self.get_logger().info(f"Yapping: {message}")
            self.yapper.yap(message)

    
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
        # convert bird_names to a list where each field is one lowercase word from each bird name
        bird_words = [word.lower() for bird in self.bird_names for word in bird.split()]
        # Remove words that appear in more than one bird name completely. Don't want them in the list even once
        bird_words = [word for word in bird_words if sum(word in b.lower() for b in self.bird_names) == 1]
        
        # Check if any of the bird words are in the text
        for word in bird_words:
            if word in text.lower():
                # Find the full bird name that contains this word
                for bird in self.bird_names:
                    if word in bird.lower():
                        self.get_logger().info(f"Found bird name '{bird}' in text: {text}")
                        return bird
        
        self.get_logger().info(f"No bird name found in text: {text}")
        return None
    
    def execute_ring_behavior(self, task):
        # First get the ring position from the description
        description = task['description']
        description_split = description.split('|')
        if len(description_split) < 3:
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
        
        color = description_split[0].strip().lower()
        
        # Rotate to face the ring
        self.get_logger().info(f"Rotating to face ring at ({x}, {y})")
        
        self.rotate_towards_point(x, y)
        time.sleep(1.0)
        
        image = self.latest_arm_image.copy()
        if image is None:
            self.get_logger().error("No image received from arm camera")
            return
        
        # Put image through AI model to detect bird species
        bird_name = predict_bird_name(image)
        
        # Replace underscores with spaces and set to lowercase
        bird_name = bird_name.replace('_', ' ').lower()
        # Check if the bird name is in the list of known birds
        if bird_name not in self.bird_names:
            self.get_logger().error(f"Detected bird species '{bird_name}' is not in the known list.")
            return
        
        self.get_logger().info(f"Detected bird species: {bird_name}")
        
        # Add the bird name and image AND NOTHING ELSE to the bird catalogue
        if bird_name not in self.bird_catalogue:
            self.bird_catalogue[bird_name] = image
            self.get_logger().info(f"Added {bird_name} to bird catalogue.")
        
        # Add the bird name to the photographed_birds dictionary by making an object with pose and color
        if bird_name not in self.photographed_birds:
            self.photographed_birds[bird_name] = {
                'pose': task['pose'],
                'color': color,
            }
        else:
            self.get_logger().warn(f"Already photographed {bird_name}.")
        
        
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
