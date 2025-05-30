#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import Point
from sensor_msgs_py import point_cloud2 as pc2
from dis_tutorial3.msg import FaceMsg

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math

from ultralytics import YOLO
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
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

		marker_topic = "/people_marker"

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

		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)
		# self.camera_info_sub = self.create_subscription(CameraInfo, "/oak/stereo/camera_info", self.caminfo_cb, qos_profile_sensor_data)
		self.depth_image_sub = self.create_subscription(Image, "/oak/right/image_raw/compressedDepth", self.depth_cb, qos_profile_sensor_data)
		self.create_subscription(CameraInfo,
								'/oak/rgb/camera_info',
								self.caminfo_cb,
								qos_profile_sensor_data)
  

		self.bridge = CvBridge()
		self.K_inv = None
		self.camera_frame = None
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)
		self.coord_pub = self.create_publisher(FaceMsg, "/face_coordinates", QoSReliabilityPolicy.BEST_EFFORT)

		self.model = YOLO("yolov8n.pt")

		self.faces = []
		self.offset = 60 # offset for point cloud access

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
		self.camera_frame = info.header.frame_id
		self.get_logger().info(f"Camera intrinsics received, frame = {self.camera_frame}")



	def depth_cb(self, img_msg: Image):
		"""
		For each detected face:
		- read z = depth[v,u] at center and at each corner
		- back-project all five points
		- TF into base_link
		- publish Marker and FaceMsg (with corners)
		"""
		if self.K_inv is None or not self.faces:
			return

		depth = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='32FC1')
		h, w = depth.shape

		for idx, (u, v, tl, br, tr, bl) in enumerate(self.faces):
			# helper to back-project a single pixel
			def unproject(px, py):
				if px<0 or px>=w or py<0 or py>=h:
					return None
				z = float(depth[py, px])
				if z <= 0.0:
					return None
				p = np.array([px, py, 1.0])
				X, Y, Z = z * (self.K_inv @ p)
				return (X, Y, Z)

			pts_cam = {
				'center': unproject(u, v),
				'tl'    : unproject(*tl),
				'br'    : unproject(*br),
				'tr'    : unproject(*tr),
				'bl'    : unproject(*bl),
			}
			if any(pt is None for pt in pts_cam.values()):
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
															'base_link',
															timeout_sec=0.5)
				except Exception as e:
					self.get_logger().warn(f"TF failed for {key}: {e}")
					stamped = {}
					break
			if len(stamped) < 5:
				continue

			# publish a sphere marker at the center
			m = Marker()
			m.header = stamped['center'].header
			m.ns, m.id, m.type = "faces", idx, Marker.SPHERE
			m.scale.x = m.scale.y = m.scale.z = 0.1
			m.color.r = 1.0; m.color.a = 1.0
			m.pose.position = stamped['center'].point
			self.marker_pub.publish(m)

			# publish FaceMsg with all five points
			fm = FaceMsg()
			fm.target_pose.header = stamped['center'].header
			fm.target_pose.pose.position = stamped['center'].point

			fm.top_left_point     = stamped['tl'].point
			fm.bottom_right_point = stamped['br'].point
			fm.top_right_point    = stamped['tr'].point
			fm.bottom_left_point  = stamped['bl'].point

			self.coord_pub.publish(fm)

	def rgb_callback(self, img_msg: Image):
		"""
		1) Run YOLO on the incoming RGB image.
		2) For each detection, compute center + the four corners pulled in by 30%.
		3) Store (u,v, tl, br, tr, bl) in self.faces.
		"""
		cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
		res = self.model.predict(cv_img,
								imgsz=(256,320),
								show=False, verbose=False,
								classes=[0],
								device=self.device)

		self.faces.clear()
		for det in res:
			if det.boxes.xyxy.nelement() == 0:
				continue
			x1,y1,x2,y2 = det.boxes.xyxy[0].tolist()
			# center
			u = int((x1 + x2)/2)
			v = int((y1 + y2)/2)

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

			self.faces.append((u, v, tl, br, tr, bl))





	def rgb_callback(self, data):

		self.faces = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

			self.get_logger().info(f"Running inference on image...")

			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)
   
			# Draw detection threshold (offset)
			width = data.width
			height = data.height
			cv_image = cv2.rectangle(cv_image, (self.offset, self.offset), (width-self.offset, height-self.offset), (0, 255, 0), 2)

			# iterate over results
			for x in res:
				bbox = x.boxes.xyxy
				if bbox.nelement() == 0: # skip if empty
					continue

				self.get_logger().info(f"Person has been detected!")

				bbox = bbox[0]

				# draw rectangle
				cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)

				cx = int((bbox[0]+bbox[2])/2)
				cy = int((bbox[1]+bbox[3])/2)

				# draw the center of bounding box
				cv_image = cv2.circle(cv_image, (cx,cy), 5, self.detection_color, -1)
    
				# Draw these in blue
				center_point = np.array([cx, cy])
				top_left_point = np.array([int(bbox[0]), int(bbox[1])])
				bottom_right_point = np.array([int(bbox[2]), int(bbox[3])])
				top_right_point = np.array([int(bbox[2]), int(bbox[1])])
				bottom_left_point = np.array([int(bbox[0]), int(bbox[3])])
    
    
				# Pull points X% closer to the center point
				percentage = 0.3
				top_left_point = top_left_point + percentage * (center_point - top_left_point)
				bottom_right_point = bottom_right_point + percentage * (center_point - bottom_right_point)
				top_right_point = top_right_point + percentage * (center_point - top_right_point)
				bottom_left_point = bottom_left_point + percentage * (center_point - bottom_left_point)
    
				# Draw these points in different colors
				cv_image = cv2.circle(cv_image, tuple(top_left_point.astype(int)), 5, (255, 0, 0), -1)
				cv_image = cv2.circle(cv_image, tuple(bottom_right_point.astype(int)), 5, (0, 255, 0), -1)
				cv_image = cv2.circle(cv_image, tuple(top_right_point.astype(int)), 5, (0, 0, 255), -1)
				cv_image = cv2.circle(cv_image, tuple(bottom_left_point.astype(int)), 5, (255, 255, 0), -1)


				self.faces.append((cx,cy, top_left_point, bottom_right_point, top_right_point, bottom_left_point))

			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key==27:
				print("exiting")
				exit()
			
		except CvBridgeError as e:
			print(e)

	def pointcloud_callback(self, data):

		# get point cloud attributes
		height = data.height
		width = data.width
		point_step = data.point_step
		row_step = data.row_step		

		# iterate over face coordinates
		for x,y, top_left, bottom_right, top_right, bottom_left in self.faces:

			# get 3-channel representation of the poitn cloud in numpy format
			a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
			a = a.reshape((height,width,3))

			# read center coordinates
			if x-self.offset < 0 or x+self.offset >= width or y-self.offset < 0 or y+self.offset >= height:
				self.get_logger().warn(f"Skipping face at ({x},{y}) due to out of bounds access.")
				continue
   
			d = a[y,x,:]
			top_left = (int(top_left[0]), int(top_left[1]))
			bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
			top_right = (int(top_right[0]), int(top_right[1]))
			bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
			tl = a[top_left[1],top_left[0],:]
			br = a[bottom_right[1],bottom_right[0],:]
			tr = a[top_right[1],top_right[0],:]
			bl = a[bottom_left[1],bottom_left[0],:]

			# create marker
			marker = Marker()

			marker.header.frame_id = "base_link"
			marker.header.stamp = data.header.stamp

			marker.type = 2
			marker.id = 0

			# Set the scale of the marker
			scale = 0.1
			marker.scale.x = scale
			marker.scale.y = scale
			marker.scale.z = scale

			# Set the color
			marker.color.r = 1.0
			marker.color.g = 1.0
			marker.color.b = 1.0
			marker.color.a = 1.0

			# Set the pose of the marker
			marker.pose.position.x = float(d[0])
			marker.pose.position.y = float(d[1])
			marker.pose.position.z = float(d[2])

			self.marker_pub.publish(marker)
   

			# create face message
			face_msg = FaceMsg()

			# 1) Fill in the stamped target pose
			face_msg.target_pose.header.frame_id = "base_link"
			face_msg.target_pose.header.stamp = data.header.stamp   # or use self.get_clock().now().to_msg()

			# say you computed px,py,pz and qx,qy,qz,qw somehow:
			face_msg.target_pose.pose.position.x = float(d[0])
			face_msg.target_pose.pose.position.y = float(d[1])
			face_msg.target_pose.pose.position.z = float(d[2])

			# 2) Fill in your left_point
			face_msg.bottom_left_point.x = float(bl[0])
			face_msg.bottom_left_point.y = float(bl[1])
			face_msg.bottom_left_point.z = float(bl[2])
			# 3) Fill in your right_point
			face_msg.bottom_right_point.x = float(br[0])
			face_msg.bottom_right_point.y = float(br[1])
			face_msg.bottom_right_point.z = float(br[2])
			# 4) Fill in your top_point
			face_msg.top_left_point.x = float(tl[0])
			face_msg.top_left_point.y = float(tl[1])
			face_msg.top_left_point.z = float(tl[2])
			# 5) Fill in your top_point
			face_msg.top_right_point.x = float(tr[0])
			face_msg.top_right_point.y = float(tr[1])
			face_msg.top_right_point.z = float(tr[2])

			# finally, publish it
			self.coord_pub.publish(face_msg)

def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()