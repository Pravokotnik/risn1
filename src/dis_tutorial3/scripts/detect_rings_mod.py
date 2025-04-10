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
from std_msgs.msg import ColorRGBA, String
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from collections import Counter

from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')

        # Basic ROS stuff
        timer_frequency = 2
        timer_period = 1/timer_frequency

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # Marker array object used for visualizations
        self.marker_array = MarkerArray()
        self.marker_num = 1

        self.latest_rgb_image = None

        # Subscribe to the image and/or depth topic
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.image_callback, 1)
        self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, 1)

        # Publiser for the visualization markers
        self.marker_pub = self.create_publisher(Marker, "/ring_marker", QoSReliabilityPolicy.BEST_EFFORT)
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

        # Object we use for transforming between coordinate frames
        # self.tf_buf = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buf)   

        # publisher za barvo
        self.color_publisher = self.create_publisher(String, "/ring/color", 10)

        # za bouding box
        self.ring_centers = [] 

    def rgb_callback(self, data):
        try:
            self.latest_rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
    

    def image_callback(self, data):
        #self.get_logger().info(f"I got a new image! Will try to find rings...")

        self.ring_centers = []

        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)

        depth_image[depth_image==np.inf] = 0
        
        # Do the necessairy conversion so we can visuzalize it in OpenCV
        image_1 = depth_image / 65536.0 * 255
        image_1 = image_1/np.max(image_1)*255

        image_viz = np.array(image_1, dtype= np.uint8)

        cv2.imshow("Depth window", image_viz)
        cv2.waitKey(1)

        process_img = image_viz.copy()
        process_img[process_img < 1] = 0
        process_img[process_img >= 1] = 255

        edges = cv2.Canny(image_viz, 50, 150)
        cv2.imshow("Edge Image", edges)
        cv2.waitKey(1)

        cv2.imshow("Binary Image", process_img)
        cv2.waitKey(1)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cim = cv2.cvtColor(image_viz, cv2.COLOR_GRAY2BGR)
        imv = cim.copy()
        cv2.drawContours(cim, contours, -1, (255, 0, 0), 1)
        cv2.imshow("Detected contours", cim)
        cv2.waitKey(1)


        elps = []
        for cnt in contours:
            #     print cnt
            #     print cnt.shape
            if cnt.shape[0] >= 20:
                ellipse = cv2.fitEllipse(cnt)
                elps.append(ellipse)

        candidates = []
        for n in range(len(elps)):
            for m in range(n + 1, len(elps)):
                # e[0] is the center of the ellipse (x,y), e[1] are the lengths of major and minor axis (major, minor), e[2] is the rotation in degrees
                
                e1 = elps[n]
                e2 = elps[m]
                dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
                angle_diff = np.abs(e1[2] - e2[2])

                # The centers of the two elipses should be within 5 pixels of each other (is there a better treshold?)
                if dist >= 5:
                    continue

                # The rotation of the elipses should be whitin 4 degrees of eachother
                if angle_diff>4:
                    continue

                e1_minor_axis = e1[1][0]
                e1_major_axis = e1[1][1]

                e2_minor_axis = e2[1][0]
                e2_major_axis = e2[1][1]

                if e1_major_axis>=e2_major_axis and e1_minor_axis>=e2_minor_axis: # the larger ellipse should have both axis larger
                    le = e1 # e1 is larger ellipse
                    se = e2 # e2 is smaller ellipse
                elif e2_major_axis>=e1_major_axis and e2_minor_axis>=e1_minor_axis:
                    le = e2 # e2 is larger ellipse
                    se = e1 # e1 is smaller ellipse
                else:
                    continue # if one ellipse does not contain the other, it is not a ring
                
                # # The widths of the ring along the major and minor axis should be roughly the same
                # border_major = (le[1][1]-se[1][1])/2
                # border_minor = (le[1][0]-se[1][0])/2
                # border_diff = np.abs(border_major - border_minor)

                # if border_diff>4:
                #     continue
                    
                candidates.append((e1,e2))

        #print("Processing is done! found", len(candidates), "candidates for rings")

        # Plot the rings on the image
        for c in candidates:

            # the centers of the ellipses
            e1 = c[0]
            e2 = c[1]

            # drawing the ellipses on the image
            cv2.ellipse(imv, e1, (0, 255, 0), 2)
            cv2.ellipse(imv, e2, (0, 255, 0), 2)

            # Get a bounding box, around the first ellipse ('average' of both elipsis)
            size = (e1[1][0]+e1[1][1])/2
            center = (e1[0][1], e1[0][0])

            x1 = int(center[0] - size / 2)
            x2 = int(center[0] + size / 2)
            x_min = x1 if x1>0 else 0
            x_max = x2 if x2<imv.shape[0] else imv.shape[0]

            y1 = int(center[1] - size / 2)
            y2 = int(center[1] + size / 2)
            y_min = y1 if y1 > 0 else 0
            y_max = y2 if y2 < imv.shape[1] else imv.shape[1]

        if len(candidates)>0:
            
            rgb_img = self.latest_rgb_image.copy()
            cv2.imshow("RGB Image", rgb_img)
            rings_rgb = rgb_img.copy()

            rings_depth = image_viz.copy()
            rings_depth = cv2.cvtColor(rings_depth, cv2.COLOR_GRAY2BGR) 

            for c in candidates:
                # centri elips
                e1 = c[0]
                e2 = c[1]

                # risemo elipse
                cv2.ellipse(rings_rgb, e1, (0, 255, 0), 2)
                cv2.ellipse(rings_rgb, e2, (0, 255, 0), 2)

                # BOUDNING BOX ######
                angles = [0,10,20]  # in degrees
                ring_points = []
                outer_ellipse = e1 if (e1[1][0] * e1[1][1]) > (e2[1][0] * e2[1][1]) else e2

                for angle in angles:
                    # Convert angle to radians
                    angle_rad = np.radians(angle)
                    
                    # Calculate point on ellipse
                    a = outer_ellipse[1][0] / 2  # semi-major axis
                    b = outer_ellipse[1][1] / 2  # semi-minor axis
                    rot = np.radians(outer_ellipse[2])
                    
                    # Parametric equations for ellipse
                    x = outer_ellipse[0][0] + a * np.cos(angle_rad) * np.cos(rot) - b * np.sin(angle_rad) * np.sin(rot)
                    y = outer_ellipse[0][1] + a * np.cos(angle_rad) * np.sin(rot) + b * np.sin(angle_rad) * np.cos(rot)

                    # check if nan
                    if np.isnan(x) or np.isnan(y):
                        continue
                    
                    ring_points.append((int(x), int(y)))
                    
                    # Draw the points (for visualization)
                    cv2.circle(rings_rgb, (int(x), int(y)), 3, (0, 0, 255), -1)
                    cv2.circle(rings_depth, (int(x), int(y)), 3, (0, 0, 255), -1)

                self.ring_points = ring_points  # This replaces self.ring_centers
                #################

                center_x = int(e1[0][0])
                center_y = int(e1[0][1])

                ring_color = self.detect_ring_color(rgb_img, center_x, center_y, e1, e2)
                self.visualize_ring_color_detection(rgb_img, center_x, center_y, e1, e2, ring_color)

                # dobim circularity
                score1 = self.compute_circularity_score(e1)
                score2 = self.compute_circularity_score(e2)
                avg_score = (score1 + score2) / 2.0


                self.publish_ring(ring_color, avg_score)


            cv2.imshow("Detected rings",imv)
            cv2.waitKey(1)

            cv2.imshow("RGB Rings", rings_rgb)
            cv2.waitKey(1)

    def detect_ring_color(self, rgb_image, center_x, center_y, e1, e2):

        if e1[1][0] * e1[1][1] > e2[1][0] * e2[1][1]:
            outer_ellipse = e1
            inner_ellipse = e2
        else:
            outer_ellipse = e2
            inner_ellipse = e1

        # maske
        height, width = rgb_image.shape[:2]
        outer_mask = np.zeros((height, width), dtype=np.uint8)
        inner_mask = np.zeros((height, width), dtype=np.uint8)

        cv2.ellipse(outer_mask, outer_ellipse, 255, -1)
        cv2.ellipse(inner_mask, inner_ellipse, 255, -1)

        ring_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))

        # operacije
        kernel = np.ones((3, 3), np.uint8)
        ring_mask = cv2.morphologyEx(ring_mask, cv2.MORPH_OPEN, kernel)

        # Visualization with gray background instead of black
        gray_background = np.full_like(rgb_image, 127)
        masked_image = gray_background.copy()
        masked_image[ring_mask > 0] = rgb_image[ring_mask > 0]
        cv2.imshow("Masked Ring", masked_image)
        cv2.waitKey(1)

        # pixli ringov
        ring_pixels = []
        y_coords, x_coords = np.where(ring_mask > 0)
        for y, x in zip(y_coords, x_coords):
            ring_pixels.append(rgb_image[y, x])

        ring_pixels = np.array(ring_pixels)

        if len(ring_pixels) < 20:
            return "not enough pixels"

        # HSV
        hsv_pixels = cv2.cvtColor(np.array([ring_pixels]), cv2.COLOR_BGR2HSV)[0]

        h_avg = np.median(hsv_pixels[:, 0])
        s_avg = np.median(hsv_pixels[:, 1])
        v_avg = np.median(hsv_pixels[:, 2])

        if v_avg < 60:
            return "black"
        if s_avg > 40:
            if h_avg < 10 or h_avg > 170:
                return "red"
            elif 35 <= h_avg < 80:
                return "green"
            elif 80 <= h_avg < 130:
                return "blue"
            else:
                return "unknown"
        else:
            return "unknown"


    def visualize_ring_color_detection(self, rgb_image, center_x, center_y, e1, e2, detected_color):
        # slika za vizualizacijo
        viz_image = rgb_image.copy()
        
        # Determine inner and outer ellipses
        if e1[1][0] * e1[1][1] > e2[1][0] * e2[1][1]:
            outer_ellipse = e1
            inner_ellipse = e2
        else:
            outer_ellipse = e2
            inner_ellipse = e1

        # color map za vsako detected barvo
        color_map = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "black": (0, 0, 0),
        }

        box_color = color_map.get(detected_color, (128, 128, 128))  # Default to gray if color not in map
    
        # bounding box
        half_width = int(outer_ellipse[1][0] / 2)
        half_height = int(outer_ellipse[1][1] / 2)
        
        x1 = int(outer_ellipse[0][0] - half_width)
        y1 = int(outer_ellipse[0][1] - half_height)
        
        x2 = int(outer_ellipse[0][0] + half_width)
        y2 = int(outer_ellipse[0][1] + half_height)
        
        height, width = rgb_image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)
        
        # narisi bounding box
        cv2.rectangle(viz_image, (x1, y1), (x2, y2), box_color, 2)
        
        # Display the visualization
        cv2.imshow("Ring Color Detection", viz_image)
        cv2.waitKey(1)

    def compute_circularity_score(self, ellipse):
        minor = min(ellipse[1])
        major = max(ellipse[1])
        if major == 0:
            return 0.0
        return minor / major
        
        
    def publish_ring(self, color, circularity):
        """Publish the detected color to a ROS topic"""
        # Optional: Add a cooldown mechanism
        if hasattr(self, 'last_published_color') and color == self.last_published_color:
            if hasattr(self, 'publish_cooldown'):
                self.publish_cooldown -= 1
                if self.publish_cooldown > 0:
                    return
        
        # Reset cooldown and store the new color
        self.last_published_color = (color, circularity)
        self.publish_cooldown = 5  # Wait 5 frames before publishing the same color again
        
        # Create and publish the message
        msg = String()
        msg.data = f"color={color}, circularity={circularity:.3f}"
        self.color_publisher.publish(msg)
        self.get_logger().info(f"Published ring color: {color}")
    
    def pointcloud_callback(self, data):
        if not hasattr(self, 'ring_points') or len(self.ring_points) == 0:
            return

        height = data.height
        width = data.width
        
        # Get 3D points for all ring points
        a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
        a = a.reshape((height, width, 3))
        
        markers = []
        
        for i, (x, y) in enumerate(self.ring_points):
            # Make sure coordinates are within bounds
            if 0 <= y < height and 0 <= x < width:
                d = a[y, x, :]
                
                marker = Marker()
                marker.header.frame_id = "/base_link"
                marker.header.stamp = data.header.stamp
                marker.type = 2  # Sphere
                marker.id = i  # Unique ID for each point
                
                # Set the scale of the marker
                scale = 0.05  # Smaller than before since we have multiple points
                marker.scale.x = scale
                marker.scale.y = scale
                marker.scale.z = scale
                
                # Different colors for different points
                marker.color.r = float(i == 0)  # Red for first point
                marker.color.g = float(i == 1)  # Green for second point
                marker.color.b = float(i == 2)  # Blue for third point
                marker.color.a = 1.0
                
                # Set the pose of the marker
                marker.pose.position.x = float(d[0])
                marker.pose.position.y = float(d[1])
                marker.pose.position.z = float(d[2])
                
                markers.append(marker)
                
        
        # Publish all markers
        for marker in markers:
            self.get_logger().info(f"Publishing marker ID: {marker.id} at position: ({marker.pose.position.x}, {marker.pose.position.y}, {marker.pose.position.z})")
            self.marker_pub.publish(marker)
        

def main():

    rclpy.init(args=None)
    rd_node = RingDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()