#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class RingDetector(Node):
    def __init__(self):
        super().__init__('depth_ring_detector')

        self.bridge = CvBridge()

        # Subscribe only to depth image
        self.depth_sub = self.create_subscription(
            Image, "/oakd/rgb/preview/depth", self.depth_callback, 1
        )

        # OpenCV visualization windows
        cv2.namedWindow("Depth raw", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Binary depth", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)

    def depth_callback(self, data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)
            return

        # Clean up and normalize
        depth_image[depth_image == np.inf] = 0
        depth_image = np.nan_to_num(depth_image)

        # Scale depth to 8-bit image for visualization and processing
        max_val = np.max(depth_image)
        if max_val == 0:
            return  # avoid division by zero
        depth_scaled = (depth_image / max_val * 255).astype(np.uint8)

        cv2.imshow("Depth raw", depth_scaled)

        # Blur and threshold to highlight potential ring shapes
        blurred = cv2.GaussianBlur(depth_scaled, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        cv2.imshow("Binary depth", thresh)

        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        ring_candidates = []
        output_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        for i, cnt in enumerate(contours):
            if len(cnt) < 20:
                continue

            if hierarchy[0][i][3] != -1:
                continue  # ignore child contours

            area = cv2.contourArea(cnt)
            if area < 200 or area > 10000:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.7:
                continue

            try:
                ellipse = cv2.fitEllipse(cnt)
            except:
                continue

            center = ellipse[0]
            size = ellipse[1]
            major_axis = max(size)
            minor_axis = min(size)
            aspect_ratio = major_axis / minor_axis

            if aspect_ratio > 1.5 or major_axis > 200:
                continue

            ring_candidates.append(ellipse)
            cv2.ellipse(output_image, ellipse, (0, 255, 0), 2)


        self.get_logger().info(f"Found {len(ring_candidates)} depth ring candidate(s)")
        cv2.imshow("Detected rings", output_image)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = RingDetector()
    rclpy.spin(node)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
