#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2, CameraInfo, CompressedImage
from geometry_msgs.msg import Point
from sensor_msgs_py import point_cloud2 as pc2
from dis_tutorial3.msg import FaceMsg

from visualization_msgs.msg import Marker, MarkerArray

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math

# Replace YOLO with DeepFace
from deepface import DeepFace
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs
from tf2_ros import Buffer, TransformListener


# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

class detect_faces(Node):

	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		marker_topic = "waypoints"

		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()
		self.scan = None

		self.create_subscription(
			Image,
			"/oak/rgb/image_raw",
			self.rgb_callback,
			qos_profile_sensor_data
		)

		# self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)
		# self.camera_info_sub = self.create_subscription(CameraInfo, "/oak/stereo/camera_info", self.caminfo_cb, qos_profile_sensor_data)
		self.depth_image_sub = self.create_subscription(CompressedImage, "/oak/stereo/image_raw/compressedDepth", self.depth_cb, qos_profile_sensor_data)
		self.create_subscription(CameraInfo,
								'/oak/rgb/camera_info',
								self.caminfo_cb,
								qos_profile_sensor_data)
  

		self.K_inv = None
		self.camera_frame = None
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.marker_pub = self.create_publisher(MarkerArray, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)
		self.coord_pub = self.create_publisher(FaceMsg, "/face_coordinates", QoSReliabilityPolicy.BEST_EFFORT)

		# No need to initialize DeepFace with a model

		self.faces = []
		self.top_offset = 100 # offset for point cloud access
		self.bottom_offset = 60 # offset for point cloud access
		self.left_offset = 60 # offset for point cloud access
		self.right_offset = 60 # offset for point cloud access
		self.gotK = False

		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")
	
 
 
 
	# Try with oak/points
	from sensor_msgs.msg import PointCloud2
	from sensor_msgs_py import point_cloud2 as pc2
	def cloud_cb(self, data: PointCloud2):
		# read into an (H×W×3) numpy array
		arr = pc2.read_points_numpy(data, field_names=('x','y','z'))
		pts = arr.reshape((data.height, data.width, 3))
		# for each detected face at (u,v):
		X, Y, Z = pts[v, u]   # already in camera frame, in meters
		# then tf2-transform into base_link if you need
  

	# Stereo camera --------------------------------------------------------------------
	# Callback for stereo camera info
	def caminfo_cb(self, info: CameraInfo):
		"""
		Grab and invert the 3×3 intrinsic matrix once, and remember the camera frame.
		"""
		K = np.array(info.k).reshape(3,3)
		self.K_inv = np.linalg.inv(K)
		self.camera_frame = "oakd_rgb_camera_optical_frame"
		if not self.gotK:
			self.get_logger().info(f"Camera intrinsic matrix K: {K}")
			self.get_logger().info(f"Camera frame: {self.camera_frame}")
			self.gotK = True



	def depth_cb(self, img_msg: CompressedImage):
		try:
			# Extract raw data from compressed message
			data = bytes(img_msg.data)
			
			# Extract header information
			header_size = 12  # Based on your message structure
			if len(data) < header_size:
				self.get_logger().error("Compressed depth data too short")
				return
				
			# Skip first 12 bytes (0,0,0,0 + 255,255,0,0 + 48,201,180,114)
			# These appear to be metadata or format indicators
			png_data = data[header_size:]
			
			# Decode PNG using OpenCV
			depth16 = cv2.imdecode(np.frombuffer(png_data, np.uint8), cv2.IMREAD_UNCHANGED)
			
			if depth16 is None:
				return

			# Process depth image
			# depth16 = cv2.medianBlur(depth16, 5)
			depth_image = depth16.astype(np.float32) / 1000.0  # Convert to meters
   
			cv2.imshow("Depth", depth_image)
			key = cv2.waitKey(1)
			
			# Only process if we have faces and camera info
			if not self.faces or self.K_inv is None:
				return

			h, w = depth_image.shape
			for idx, (u, v, tl, br, tr, bl) in enumerate(self.faces):
				# helper to back-project a single pixel
				def unproject(px, py):
					if not (0 <= px < w and 0 <= py < h):
						return None
					z = depth_image[py, px]
					self.get_logger().info(f"Depth at ({px},{py}): {z}")
					if z <= 0.1:  # Minimum valid depth threshold
						return None
					p = np.array([px, py, 1.0])
					return z * (self.K_inv @ p)

				pts_cam = {
					'center': unproject(u, v),
					'tl'    : unproject(*tl),
					'br'    : unproject(*br),
					'tr'    : unproject(*tr),
					'bl'    : unproject(*bl),
				}
				if any(pt is None for pt in pts_cam.values()):
					self.get_logger().warn(f"Skipping face {idx} due to invalid depth data.")
					continue

				# TF each into base_link
				stamped = {}
				for key,(X,Y,Z) in pts_cam.items():
					ps = PointStamped()
					ps.header.frame_id = self.camera_frame
					ps.header.stamp    = img_msg.header.stamp
					ps.point.x, ps.point.y, ps.point.z = X, Y, Z
					try:
						stamped[key] = self.tf_buffer.transform(ps,
																'base_link')
					except Exception as e:
						self.get_logger().warn(f"TF failed for {key}: {e}")
						stamped = {}
						break
				if len(stamped) < 5:
					self.get_logger().warn(f"Skipping face {idx} due to incomplete transformation.")
					continue
    
				# Instead of transforming, we can just use the PointStamped directly
				# stamped = {}
				# for key, (X, Y, Z) in pts_cam.items():
				# 	ps = PointStamped()
				# 	ps.header.frame_id = "base_link"  # Use base_link directly
				# 	ps.header.stamp    = img_msg.header.stamp
				# 	ps.point.x, ps.point.y, ps.point.z = X, Y, Z
				# 	stamped[key] = ps

				# publish a sphere marker at the center
				m = Marker()
				m.header = stamped['center'].header
				m.ns, m.id, m.type = "faces", idx, Marker.SPHERE
				m.scale.x = m.scale.y = m.scale.z = 0.1
				m.color.r = 1.0; m.color.a = 1.0
				m.pose.position = stamped['center'].point
    
				# Convert to MarkerArray
				marker_array = MarkerArray()
				marker_array.markers.append(m)
				self.marker_pub.publish(marker_array)

				# publish FaceMsg with all five points
				fm = FaceMsg()
				fm.target_pose.header = stamped['center'].header
				fm.target_pose.pose.position = stamped['center'].point

				fm.top_left_point     = stamped['tl'].point
				fm.bottom_right_point = stamped['br'].point
				fm.top_right_point    = stamped['tr'].point
				fm.bottom_left_point  = stamped['bl'].point

				self.coord_pub.publish(fm)
				self.get_logger().info(f"Published face {idx} with center at ({stamped['center'].point.x}, ")

			
		except Exception as e:
			self.get_logger().error(f"Error processing depth image: {e}")
			return

	def rgb_callback(self, img_msg: Image):
		"""
		1) Run DeepFace on the incoming RGB image.
		2) For each detection, compute center + the four corners pulled in by 30%.
		3) Store (u,v, tl, br, tr, bl) in self.faces.
		"""
		try:
			cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
		
			# Use DeepFace for face detection
			faces = DeepFace.extract_faces(cv_image, 
										  detector_backend='opencv', 
										  enforce_detection=False,
										  align=False)
			
			self.faces.clear()

			# Draw detection threshold square (offset)
			width = img_msg.width
			height = img_msg.height
			cv_image = cv2.rectangle(cv_image,
									(self.left_offset, self.top_offset),
									(width - self.right_offset, height - self.bottom_offset),
									(0, 255, 0), 2)

			for face in faces:
				facial_area = face['facial_area']
				x1, y1, x2, y2 = facial_area['x'], facial_area['y'], facial_area['x'] + facial_area['w'], facial_area['y'] + facial_area['h']
				
				if x1 < 5 and x2 > width - 5  and y1 < 5 and y2 > height - 5:
					continue
    
				# center
				u = int((x1 + x2)/2)
				v = int((y1 + y2)/2)

				if v < self.top_offset or v >= height - self.bottom_offset or \
				   u < self.left_offset or u >= width - self.right_offset:
					self.get_logger().warn(f"Skipping face at ({u},{v}) due to out of bounds access.")
					continue

				# raw corners
				tl = np.array([x1, y1])
				br = np.array([x2, y2])
				tr = np.array([x2, y1])
				bl = np.array([x1, y2])

				# pull 30% toward center
				pct = 0.3
				center = np.array([u, v])
				tl = tl + pct*(center - tl)
				br = br + pct*(center - br)
				tr = tr + pct*(center - tr)
				bl = bl + pct*(center - bl)

				# round to ints
				tl = tl.astype(int)
				br = br.astype(int)
				tr = tr.astype(int)
				bl = bl.astype(int)

				cv_image = cv2.circle(cv_image, tuple(tl), 5, (255, 0, 0), -1)
				cv_image = cv2.circle(cv_image, tuple(br), 5, (0, 255, 0), -1)
				cv_image = cv2.circle(cv_image, tuple(tr), 5, (0, 0, 255), -1)
				cv_image = cv2.circle(cv_image, tuple(bl), 5, (255, 255, 0), -1)
				cv_image = cv2.circle(cv_image, (u, v), 5, (0, 0, 0), -1)

				self.faces.append((u, v, tl, br, tr, bl))

			cv2.imshow("RGB and Face", cv_image)
			cv2.waitKey(1)
		except Exception as e:
			self.get_logger().error(f"Error processing RGB image: {e}")
			return


def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()