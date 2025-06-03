#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker
from sensor_msgs_py import point_cloud2 as pc2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer

qos_profile = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1
)

class RingDetector(Node):
    def __init__(self):
        super().__init__('ring_detector')

        self.bridge = CvBridge()
        self.latest_rgb_image = None
        self.latest_rgb_image_msg = None
        self.ring_points = []
        self.last_published_color = None
        self.publish_cooldown = 0

        # self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)
        # self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, 1)
        self.image_sub = Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
        self.depth_sub = Subscriber(self, Image, "/oakd/rgb/preview/depth")
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile)
        self.color_pub = self.create_publisher(String, "/ring/color", 10)
        self.marker_pub = self.create_publisher(Marker, "/ring_marker", 10)
        
        # Time synchronizer to sync RGB and depth images
        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.synced_callback)
        
        self.left_right_offset = 30 # amount of pixels to crop left and right for detections
        
        
        if 'QT' in cv2.getBuildInformation():
            cv2.namedWindow("Normalized Depth Image", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
            cv2.namedWindow("Edges", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
            cv2.namedWindow("Filtered Edges", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
            cv2.namedWindow("Rings RGB Crop", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
            cv2.namedWindow("Ring Color Detection", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
            cv2.namedWindow("Ring Masked RGB", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        else:
            cv2.namedWindow("Normalized Depth Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Filtered Edges", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Rings RGB Crop", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Ring Color Detection", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Ring Masked RGB", cv2.WINDOW_NORMAL)

    # def rgb_callback(self, msg):
    #     try:
    #         self.latest_rgb_image_msg = msg
    #         self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    #     except CvBridgeError as e:
    #         self.get_logger().error(f"RGB conversion error: {e}")

    # def depth_callback(self, msg):
    #     try:
    #         depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
    #     except CvBridgeError as e:
    #         self.get_logger().error(f"Depth conversion error: {e}")
    #         return
        
    #     if self.latest_rgb_image_msg is None:
    #         self.get_logger().warn("No RGB image available for processing")
    #         return
        
    #     while msg.header.stamp != self.latest_rgb_image_msg.header.stamp:
    #         self.get_logger().warn(f"Depth: {msg.header.stamp} ||| RGB: {self.latest_rgb_image_msg.header.stamp}.")
            
    #     self.get_logger().info(f"Processing depth image at time {msg.header.stamp}")
        
    #     rgb_image = self.latest_rgb_image.copy()
        


    def synced_callback(self, rgb_msg, depth_msg):
        # Convert images from ROS msg to OpenCV
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # Now rgb_image and depth_image are synced in time, process them together
        self.process_images(rgb_image, depth_image)
    
    def process_images(self, rgb_image, depth_image):
        # Crop RGB to upper half as well

        # Filter depth range 1m to 5m
        depth_filtered = np.where((depth_image >= 1.0) & (depth_image <= 3.0), depth_image, 0)

        # Smooth depth to reduce noise while preserving edges
        depth_smooth = cv2.bilateralFilter(depth_filtered.astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=5)

        # Crop upper half (assumes rings suspended in upper half)
        h, w = depth_smooth.shape
        depth_crop = depth_smooth[0:h//2, :]
        if rgb_image is None:
            self.get_logger().warn("No RGB image available for visualization")
            return
        rgb_crop = rgb_image[0:h//2, :].copy()
        # # Offset lines
        # cv2.line(rgb_crop, (self.left_right_offset, 0), (self.left_right_offset, h//2), (0, 255, 0), 2)
        # cv2.line(rgb_crop, (rgb_crop.shape[1] - self.left_right_offset, 0), (rgb_crop.shape[1] - self.left_right_offset, h//2), (0, 255, 0), 2)
        # cv2.imshow("Rings RGB Crop", rgb_crop)
        # cv2.imshow("Ring Color Detection", rgb_image)

        # Normalize for visualization and processing
        norm_depth = cv2.normalize(depth_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        cv2.imshow("Normalized Depth Image", norm_depth)

        # Edge detection on normalized depth
        edges = cv2.Canny(norm_depth, 50, 150)
        cv2.imshow("Edges", edges)
        cv2.waitKey(1)

        # Mask out straight lines with HoughLinesP to reduce false positives
        mask_no_lines = np.ones_like(edges) * 255  # start with white mask
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=10, maxLineGap=1)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(mask_no_lines, (x1,y1), (x2,y2), 0, 3)  # draw black lines on mask

        # Combine edge image with line mask to keep edges except straight lines
        filtered_edges = cv2.bitwise_and(edges, mask_no_lines)
        
        cv2.imshow("Filtered Edges", filtered_edges)
        cv2.waitKey(1)

        # Find contours on filtered edges
        contours, _ = cv2.findContours(filtered_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # self.get_logger().info(f"Found {len(contours)} contours in depth image")

        candidates = []
        counters = [0, 0, 0, 0, 0, 0, 0]
        for cnt in contours:
            counters[0] += 1  # total contours
            if len(cnt) < 20:
                continue

            counters[1] += 1  # contours with enough points
            area = cv2.contourArea(cnt)
            # if area < 50:  # too small
            #     continue
            # if area > 5:  # too large
            #     continue

            counters[2] += 1  # contours with sufficient area
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            counters[3] += 1  # contours with valid perimeter
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            # if circularity < 0.02:  # tune threshold for ring shape
            #     continue

            counters[4] += 1  # contours with good circularity
            # Fit ellipse and check axis ratio (should not be too elongated)
            if cnt.shape[0] < 5:
                continue  # Need at least 5 points for ellipse

            counters[5] += 1  # contours with enough points for ellipse fitting
            ellipse = cv2.fitEllipse(cnt)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            axis_ratio = minor_axis / major_axis
            if axis_ratio < 0.7:  # rings shouldn't be super elongated
                continue
            
            # Ignore detections too close to the left/right edges
            center_x = int(ellipse[0][0])
            if center_x < self.left_right_offset or center_x > rgb_crop.shape[1] - self.left_right_offset:
                continue
            
            # Check area of the fitted ellipse
            # if major_axis < 10 or minor_axis < 10:  # tune thresholds for minimum size
            #     continue
            
            # if major_axis > 50 or minor_axis > 50:  # tune thresholds for maximum size
            #     continue
            self.get_logger().info(f"Detected elipse circularity: {circularity:.3f}, axis ratio: {axis_ratio:.3f}")
            counters[6] += 1  # contours with good axis ratio
            candidates.append(ellipse)
        
        # self.get_logger().info(f"Contour analysis: {counters}")
        
        ring_candidates = []
        for i, cnt in enumerate(candidates):
            # Depth hollow check
            if not self.check_depth_hollow(depth_crop, cnt):
                continue
            ring_candidates.append(cnt)  # Store both outer and inner as same for now

        # Pair candidates into possible rings (inner/outer ellipse pairs)
        # for i in range(len(candidates)):
        #     for j in range(i+1, len(candidates)):
        #         e1 = candidates[i]
        #         e2 = candidates[j]

        #         # Compute bounding box for e1
        #         x1_c, y1_c = e1[0]
        #         half_w1, half_h1 = e1[1][0] / 2, e1[1][1] / 2
        #         x1_min, x1_max = x1_c - half_w1, x1_c + half_w1
        #         y1_min, y1_max = y1_c - half_h1, y1_c + half_h1

        #         # Compute bounding box for e2
        #         x2_c, y2_c = e2[0]
        #         half_w2, half_h2 = e2[1][0] / 2, e2[1][1] / 2
        #         x2_min, x2_max = x2_c - half_w2, x2_c + half_w2
        #         y2_min, y2_max = y2_c - half_h2, y2_c + half_h2

        #         # Check if center of e1 is inside bounding box of e2 OR center of e2 inside bounding box of e1
        #         def inside_bbox(cx, cy, xmin, xmax, ymin, ymax):
        #             return xmin <= cx <= xmax and ymin <= cy <= ymax

        #         if not (inside_bbox(x1_c, y1_c, x2_min, x2_max, y2_min, y2_max) or
        #                 inside_bbox(x2_c, y2_c, x1_min, x1_max, y1_min, y1_max)):
        #             continue

        #         # Identify outer and inner ellipse by area
        #         area1 = np.pi * (e1[1][0]/2) * (e1[1][1]/2)
        #         area2 = np.pi * (e2[1][0]/2) * (e2[1][1]/2)
        #         if area1 > area2:
        #             outer, inner = e1, e2
        #         else:
        #             outer, inner = e2, e1

        #         # Depth hollow check
        #         if not self.check_depth_hollow(depth_crop, outer, inner):
        #             continue

        #         ring_candidates.append((outer, inner))
        
        # self.get_logger().info(f"Found {len(ring_candidates)} ring candidates after pairing")
        
        # If no candidates found, just return
        if len(ring_candidates) == 0:
            self.ring_points = []
            return

        # Visualization & publish
        self.get_logger().info(f"Found {len(ring_candidates)} ring candidates")

        # For each ring candidate, verify color and publish
        for ring in ring_candidates:
            # Draw ellipses on RGB crop
            cv2.ellipse(rgb_crop, ring, (0,255,0), 2)

            center_x, center_y = int(ring[0][0]), int(ring[0][1])
            ring_color = self.detect_ring_color(rgb_image, norm_depth, ring)
            self.visualize_ring_color_detection(rgb_image, ring, ring_color)

            circularity_ring = self.compute_circularity_score(ring)
            avg_circularity = (circularity_ring) / 2.0

            # Prepare ring points for pointcloud lookup: sample points on outer ellipse
            self.ring_points = self.sample_ring_points(ring, num_points=20)
            
            # Visualize the detected rings with cv2 imshow
            for (x, y) in self.ring_points:
                if 0 <= y < rgb_crop.shape[0] and 0 <= x < rgb_crop.shape[1]:
                    cv2.circle(rgb_crop, (x, y), 2, (255, 0, 0), -1)
                    
            

            self.publish_ring(ring_color, avg_circularity)


        cv2.imshow("Rings RGB Crop", rgb_crop)
        cv2.waitKey(1)

    def check_depth_hollow(self, depth_img, ring):
        # Check if the center of the ring has a distance of 0
        center_x = int(ring[0][0])
        center_y = int(ring[0][1])
        
        if center_y < 0 or center_y >= depth_img.shape[0] or center_x < 0 or center_x >= depth_img.shape[1]:
            self.get_logger().warn(f"Ring center ({center_x}, {center_y}) is out of bounds for depth image size {depth_img.shape}")
            return False
        center_depth = depth_img[center_y, center_x]
        if center_depth > 0.1:  # Tune threshold for hollow check
            return False
        return True

    def detect_ring_color(self, rgb_img, norm_depth, ring):
        h, w = rgb_img.shape[:2]
        
        # Remove any pixel of this value: #b2b2b2; from the rgb_img and make a new image from it
        no_sky_rgb = rgb_img.copy()
        no_sky_rgb[np.all(rgb_img == [178, 178, 178], axis=-1)] = [0, 0, 0]  # Set sky pixels to black

        ring_mask = np.zeros((h,w), dtype=np.uint8)
        cv2.ellipse(ring_mask, ring, 255, -1)

        # 5 pixels around the elipse
        ring_mask = cv2.dilate(ring_mask, np.ones((5,5), np.uint8), iterations=1)
        
        # # Create new mask that's 5 pixels smaller than the original and remove it from the original mask
        # ring_mask_inner = np.zeros((h,w), dtype=np.uint8)
        # cv2.ellipse(ring_mask_inner, ring, 255, -1)
        # ring_mask_inner = cv2.erode(ring_mask_inner, np.ones((3,3), np.uint8), iterations=1)
        # ring_mask = cv2.bitwise_and(ring_mask, cv2.bitwise_not(ring_mask_inner))
        
        # Keep only pixels with depth > 0.1. The norm_depth image is cut in half, so we need to add blank pixels to the bottom so it's the same size as the rgb_img
        if norm_depth.shape[0] < h:
            rows_to_add = h - norm_depth.shape[0]
            blank_rows = np.zeros((rows_to_add, w), dtype=np.uint8)
            norm_depth = np.vstack((norm_depth, blank_rows))
        ring_mask = cv2.bitwise_and(ring_mask, (norm_depth > 0.1).astype(np.uint8) * 255)
        ring_mask = cv2.dilate(ring_mask, np.ones((3,4), np.uint8), iterations=1)
        
        # Remove the sky pixels from the ring mask
        ring_mask = cv2.bitwise_and(ring_mask, cv2.cvtColor(no_sky_rgb, cv2.COLOR_BGR2GRAY))
        
        ring_masked_rgb = cv2.bitwise_and(rgb_img, rgb_img, mask=ring_mask)
        cv2.imshow("Ring Masked RGB", ring_masked_rgb)
        cv2.waitKey(1)
        ##############################################################
        # STRICT RING VERIFICATION
        cx, cy = int(ring[0][0]), int(ring[0][1])
        
        # 1. Check center is within image bounds
        if not (0 <= cx < w and 0 <= cy < h):
            return "invalid_bounds"
            
        # 2. Check center is empty (both in mask and depth)
        if ring_mask[cy, cx] != 0 or norm_depth[cy, cx] != 0:
            return "invalid_center"
        
        # UP - Check all pixels from center to top
        if np.sum(ring_mask[0:cy, cx]) == 0:
            return "invalid_up"
            
        # DOWN - Check all pixels from center to bottom
        if np.sum(ring_mask[cy+1:h, cx]) == 0:
            return "invalid_down"
            
        # LEFT - Check all pixels from center to left edge
        if np.sum(ring_mask[cy, 0:cx]) == 0:
            return "invalid_left"
            
        # RIGHT - Check all pixels from center to right edge
        if np.sum(ring_mask[cy, cx+1:w]) == 0:
            return "invalid_right"
            
        ##############################################################
        
        # If we eorde the image by X pixels, we should get nothing since the rings are thin and hollow. Non hollow objects will be too thick.
        really_dilated = cv2.erode(ring_mask, np.ones((10,10), np.uint8), iterations=1)
        if np.sum(really_dilated) > 0:
            return "invalid_thickness"



        # Extract ring pixels
        ring_pixels = rgb_img[ring_mask > 0]
        if len(ring_pixels) < 30:
            return "unknown"
        
        

        hsv_pixels = cv2.cvtColor(np.array([ring_pixels]), cv2.COLOR_BGR2HSV)[0]

        h_med = np.median(hsv_pixels[:,0])
        s_med = np.median(hsv_pixels[:,1])
        v_med = np.median(hsv_pixels[:,2])

        if v_med < 60:
            return "black"
        if s_med > 40:
            if h_med < 10 or h_med > 170:
                return "red"
            elif 35 <= h_med < 80:
                return "green"
            elif 80 <= h_med < 130:
                return "blue"
            else:
                return "unknown"
        else:
            return "unknown"

    def visualize_ring_color_detection(self, rgb_img, ring, color_name):
        viz_img = rgb_img.copy()
        color_map = {
            "red": (0,0,255),
            "green": (0,255,0),
            "blue": (255,0,0),
            "black": (0,0,0),
            # "unknown": (128,128,128)
        }
        
        # If color_name is not in the map, return
        if color_name not in color_map:
            return
        
        box_color = color_map.get(color_name, (128,128,128))

        # Bounding box around outer ellipse
        x_center, y_center = ring[0]
        half_w, half_h = ring[1][0]/2, ring[1][1]/2

        x1 = max(0, int(x_center - half_w))
        y1 = max(0, int(y_center - half_h))
        x2 = min(rgb_img.shape[1]-1, int(x_center + half_w))
        y2 = min(rgb_img.shape[0]-1, int(y_center + half_h))

        cv2.rectangle(viz_img, (x1,y1), (x2,y2), box_color, 3)
        cv2.putText(viz_img, color_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        cv2.imshow("Ring Color Detection", viz_img)
        cv2.waitKey(1)

    def compute_circularity_score(self, ellipse):
        minor = min(ellipse[1])
        major = max(ellipse[1])
        if major == 0:
            return 0.0
        return minor / major

    def sample_ring_points(self, ellipse, num_points=20):
        points = []
        center = ellipse[0]
        axes = (ellipse[1][0]/2, ellipse[1][1]/2)
        angle_deg = ellipse[2]
        angle_rad = np.radians(angle_deg)

        for theta in np.linspace(0, 2*np.pi, num_points):
            x = center[0] + axes[0] * np.cos(theta) * np.cos(angle_rad) - axes[1] * np.sin(theta) * np.sin(angle_rad)
            y = center[1] + axes[0] * np.cos(theta) * np.sin(angle_rad) + axes[1] * np.sin(theta) * np.cos(angle_rad)
            points.append((int(x), int(y)))
        return points

    def publish_ring(self, color, circularity):
        # Cooldown to avoid spamming same color
        if self.last_published_color == color and self.publish_cooldown > 0:
            self.publish_cooldown -= 1
            return

        self.last_published_color = color
        self.publish_cooldown = 5

        msg = String()
        msg.data = f"color={color}, circularity={circularity:.3f}"
        self.color_pub.publish(msg)
        self.get_logger().info(f"Published ring color: {color}")

    def pointcloud_callback(self, msg):
        if not self.ring_points or self.latest_rgb_image is None:
            return

        height = msg.height
        width = msg.width

        points_np = pc2.read_points_numpy(msg, field_names=("x","y","z"))
        points_np = points_np.reshape((height, width, 3))

        color = self.last_published_color if self.last_published_color else "unknown"

        marker = Marker()
        marker.header = msg.header
        marker.header.frame_id = "base_link"
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD

        scale = 0.05
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        color_map = {
            "red": (1.0,0.0,0.0),
            "green": (0.0,1.0,0.0),
            "blue": (0.0,0.0,1.0),
            "black": (0.0,0.0,0.0),
            "unknown": (1.0,1.0,0.0)
        }
        r,g,b = color_map.get(color, (1.0,1.0,0.0))

        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        # Compute average position of ring points
        avg_x = 0.0
        avg_y = 0.0
        avg_z = 0.0
        valid_points = 0

        marker.points = []
        for (x, y) in self.ring_points:
            if 0 <= y < height and 0 <= x < width:
                p = points_np[y,x]
                if np.any(np.isnan(p)) or np.any(np.isinf(p)):
                    continue
                avg_x += p[0]
                avg_y += p[1]
                avg_z += p[2]
                valid_points += 1

                pt = PointStamped()
                pt.point.x = float(p[0])
                pt.point.y = float(p[1])
                pt.point.z = float(p[2])
                marker.points.append(pt.point)

        if valid_points == 0:
            return

        avg_x /= valid_points
        avg_y /= valid_points
        avg_z /= valid_points

        marker.pose.position.x = avg_x
        marker.pose.position.y = avg_y
        marker.pose.position.z = avg_z

        self.marker_pub.publish(marker)
        self.get_logger().info(f"Published marker at ({avg_x:.2f},{avg_y:.2f},{avg_z:.2f}) with color {color}")

def main():
    rclpy.init()
    node = RingDetector()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
