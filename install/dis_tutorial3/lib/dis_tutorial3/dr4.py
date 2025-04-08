#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class DepthRingDetector(Node):
    def __init__(self):
        super().__init__('depth_ring_detector')

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # Marker array object used for visualizations
        self.marker_array = MarkerArray()
        self.marker_num = 1

        # Subscribe to the depth topic
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)

        # Create windows for visualization
        cv2.namedWindow("Raw Depth", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Processed Depth", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)       

    def depth_callback(self, data):
        self.get_logger().info(f"Processing depth image for ring detection")

        try:
            # Convert ROS image to OpenCV format
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)
            return

        # Step 1: Create a mask of valid depth values and invalid (inf/0) depth values
        # Invalid values are likely to be the centers of rings or areas where depth sensing fails
        invalid_mask = (depth_image == np.inf) | (depth_image == 0) | np.isnan(depth_image)
        valid_mask = ~invalid_mask
        
        # Convert to binary images for processing
        valid_binary = valid_mask.astype(np.uint8) * 255
        invalid_binary = invalid_mask.astype(np.uint8) * 255
        
        # Step 2: Find contours in the invalid regions (potential ring centers)
        invalid_contours, _ = cv2.findContours(invalid_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Normalize depth image for visualization
        depth_viz = np.zeros_like(depth_image)
        depth_viz[valid_mask] = depth_image[valid_mask]
        if np.max(depth_viz) > 0:
            depth_viz = depth_viz / np.max(depth_viz) * 255
        depth_viz = np.array(depth_viz, dtype=np.uint8)
        
        # Convert to color for visualization
        depth_color = cv2.cvtColor(depth_viz, cv2.COLOR_GRAY2BGR)
        
        # Show raw depth visualization
        cv2.imshow("Raw Depth", depth_color)
        cv2.waitKey(1)
        
        # Step 3: Process contours to find ring candidates
        ring_candidates = []
        
        for contour in invalid_contours:
            # Skip very small contours (likely noise)
            if cv2.contourArea(contour) < 20:  # Reduced minimum area for small rings
                continue
                
            # Fit an ellipse to the contour if it has enough points
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    
                    # Skip very small ellipses
                    if min(ellipse[1]) < 3:  # Reduced minimum size for small rings
                        continue
                        
                    # Skip very elongated ellipses (not likely to be rings)
                    aspect_ratio = max(ellipse[1]) / (min(ellipse[1]) + 1e-6)
                    if aspect_ratio > 2.5:  # More lenient aspect ratio
                        continue
                        
                    # Create masks for analysis
                    center_mask = np.zeros_like(valid_binary)
                    cv2.ellipse(center_mask, ellipse, 255, -1)
                    
                    # Create a slightly larger ellipse to capture the ring
                    # Using smaller expansion factor for small rings
                    outer_center = ellipse[0]
                    outer_axes = (ellipse[1][0] * 1.15, ellipse[1][1] * 1.15)  # Only 15% larger
                    outer_angle = ellipse[2]
                    outer_ellipse = (outer_center, outer_axes, outer_angle)
                    
                    # Create a mask for the outer ellipse
                    outer_mask = np.zeros_like(valid_binary)
                    cv2.ellipse(outer_mask, outer_ellipse, 255, -1)
                    
                    # Ring area is the region between inner and outer ellipses
                    ring_mask = cv2.bitwise_xor(outer_mask, center_mask)
                    
                    # Check if the ring area has enough valid depth values
                    ring_valid_ratio = np.sum(valid_mask & (ring_mask > 0)) / np.sum(ring_mask > 0) if np.sum(ring_mask > 0) > 0 else 0
                    
                    # Check if the center area has enough invalid depth values
                    center_invalid_ratio = np.sum(invalid_mask & (center_mask > 0)) / np.sum(center_mask > 0) if np.sum(center_mask > 0) > 0 else 0
                    
                    # This is a potential ring if:
                    # 1. The center is mostly empty (invalid depth values)
                    # 2. The surrounding ring area has valid depth values
                    if center_invalid_ratio > 0.6 and ring_valid_ratio > 0.4:  # Stricter ratios for better detection
                        # Calculate average depth of the ring
                        ring_depths = depth_image[valid_mask & (ring_mask > 0)]
                        avg_depth = np.mean(ring_depths) if len(ring_depths) > 0 else 0
                        
                        ring_candidates.append((ellipse, outer_ellipse, avg_depth))
                        
                        # Draw the ring on the processed depth image
                        cv2.ellipse(depth_color, ellipse, (255, 0, 0), 1)  # Inner ellipse in blue
                        cv2.ellipse(depth_color, outer_ellipse, (0, 0, 255), 1)  # Outer ellipse in red
                    
                except Exception as e:
                    # Skip if ellipse fitting fails
                    continue
        
        # Show processed depth with identified ring centers
        cv2.imshow("Processed Depth", depth_color)
        cv2.waitKey(1)
        
        # Create final visualization
        result_img = depth_color.copy()
        
        for inner_e, outer_e, depth in ring_candidates:
            # Draw center point
            center_x = int(inner_e[0][0])
            center_y = int(inner_e[0][1])
            cv2.circle(result_img, (center_x, center_y), 3, (0, 255, 0), -1)  # Green
            
            # Display depth information
            cv2.putText(result_img, f"D: {depth:.2f}m", 
                      (center_x+10, center_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        print(f"Found {len(ring_candidates)} ring candidates")
        cv2.imshow("Detected rings", result_img)
        cv2.waitKey(1)


def main():
    rclpy.init(args=None)
    rd_node = DepthRingDetector()
    rclpy.spin(rd_node)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()