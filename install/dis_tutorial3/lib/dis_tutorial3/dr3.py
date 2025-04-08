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
    depth=1
)

class RingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')

        # Basic ROS setup
        timer_frequency = 2
        timer_period = 1 / timer_frequency

        self.bridge = CvBridge()
        self.marker_array = MarkerArray()
        self.marker_num = 1

        self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)

        cv2.namedWindow("Binary Depth", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected contours (depth)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings (depth)", cv2.WINDOW_NORMAL)

    def image_callback(self, data):
        pass  # You no longer need RGB image processing

    def depth_callback(self, data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)
            return

        depth_image[np.isinf(depth_image)] = 0
        depth_image[np.isnan(depth_image)] = 0

        # Normalize and convert to 8-bit for display and processing
        normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = np.uint8(normalized)

        # Binarize the depth image
        thresh = cv2.adaptiveThreshold(depth_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
        cv2.imshow("Binary Depth", thresh)
        cv2.waitKey(1)

        # Extract contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours for visualization
        contour_img = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
        cv2.imshow("Detected contours (depth)", contour_img)
        cv2.waitKey(1)

        # Fit ellipses
        elps = []
        for cnt in contours:
            if cnt.shape[0] >= 20:
                ellipse = cv2.fitEllipse(cnt)
                elps.append(ellipse)

        candidates = []
        for n in range(len(elps)):
            for m in range(n + 1, len(elps)):
                e1 = elps[n]
                e2 = elps[m]
                dist = np.linalg.norm(np.array(e1[0]) - np.array(e2[0]))
                angle_diff = abs(e1[2] - e2[2])

                if dist >= 5 or angle_diff > 4:
                    continue

                e1_minor, e1_major = e1[1]
                e2_minor, e2_major = e2[1]

                if e1_major >= e2_major and e1_minor >= e2_minor:
                    le, se = e1, e2
                elif e2_major >= e1_major and e2_minor >= e1_minor:
                    le, se = e2, e1
                else:
                    continue

                candidates.append((le, se))

        print("Depth image processing is done! Found", len(candidates), "candidates for rings")

        # Draw the detected rings
        for e1, e2 in candidates:
            cv2.ellipse(contour_img, e1, (0, 255, 0), 2)
            cv2.ellipse(contour_img, e2, (0, 255, 0), 2)

        if candidates:
            cv2.imshow("Detected rings (depth)", contour_img)
            cv2.waitKey(1)

def main():
    rclpy.init(args=None)
    rd_node = RingDetector()
    rclpy.spin(rd_node)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
