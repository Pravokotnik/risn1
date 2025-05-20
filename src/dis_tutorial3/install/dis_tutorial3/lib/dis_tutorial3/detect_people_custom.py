#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from ultralytics import YOLO

from geometry_msgs.msg import PoseWithCovarianceStamped
import math
from collections import defaultdict

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

		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

		self.amcl_pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.amcl_pose_callback, qos_profile_sensor_data)

		self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)

		self.model = YOLO("yolov8n.pt")

		self.faces = []
		self.closest_faces = {}

		self.robot_position = None
		self.robot_yaw = None

		self.output_file = "faces_pos.txt"

		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")

	def amcl_pose_callback(self, data):
		self.robot_position = (
            data.pose.pose.position.x,
            data.pose.pose.position.y
        )

		orientation = data.pose.pose.orientation
		self.robot_yaw = self.quaternion_to_yaw(orientation)
	
	def quaternion_to_yaw(self, orientation):
		x = orientation.x
		y = orientation.y
		z = orientation.z
		w = orientation.w

		yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
		return yaw
	 
	def save_dictionary_to_file(self, face_id, robot_coords, face_coords, robot_yaw):
		robot_coords_with_yaw = (robot_coords[0], robot_coords[1], robot_yaw)
		with open(self.output_file, "a") as f:
			f.write(f"Face ID: {face_id}, Robot Coordinates: {robot_coords_with_yaw}, Face Coordinates: {face_coords}\n")
		self.get_logger().info(f"Appended new entry to {self.output_file}")
		
	def save_whole_dictionary_to_file(self):
		"""Writes the entire closest_faces dictionary to the output file with section headers"""
		with open(self.output_file, "a") as f:  # Use "a" to append to existing file
			f.write("\n============whole dict=============\n")
			for face_id, (distance, robot_coords, face_coords) in self.closest_faces.items():
				f.write(f"Face ID: {face_id}, "
						f"Robot Coordinates: {robot_coords}, "
						f"Face Coordinates: {face_coords}, "
						f"Distance: {distance:.2f}m\n")
			f.write("=================================\n")
		self.get_logger().info("Saved entire face dictionary to file")


	def rgb_callback(self, data):

		self.faces = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

			self.get_logger().info(f"Running inference on image...")

			# run inference
			res = self.model.track(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device, persist=True)

			# iterate over results
			for x in res:
				bbox = x.boxes.xyxy
				if bbox.nelement() == 0: # skip if empty
					continue

				track_id = int(x.boxes.id[0]) if x.boxes.id is not None else 0
				self.get_logger().info(f"Person {track_id} detected!")

				bbox = bbox[0]

				# draw rectangle
				cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)

				cx = int((bbox[0]+bbox[2])/2)
				cy = int((bbox[1]+bbox[3])/2)

				# draw the center of bounding box
				cv_image = cv2.circle(cv_image, (cx,cy), 5, self.detection_color, -1)

				self.faces.append((cx, cy, track_id))
				cv2.putText(cv_image, f"ID: {track_id}", (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.detection_color, 2)

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

		if self.robot_position is None or self.robot_yaw is None:
			self.get_logger().info("Skipping face detection - waiting for robot position data")
			return     

		# iterate over face coordinates
		for x,y,face_id in self.faces:

			# get 3-channel representation of the poitn cloud in numpy format
			a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
			a = a.reshape((height,width,3))

			# read center coordinates
			d = a[y,x,:]

			# create marker
			marker = Marker()

			marker.header.frame_id = "/base_link"
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

			face_coords_3d = (float(d[0]), float(d[1]), float(d[2]))
			
			current_dist = math.hypot(d[0], d[1])
			robot_x, robot_y = self.robot_position
			yaw = self.robot_yaw
			face_x_map = robot_x + d[0] * math.cos(yaw) - d[1] * math.sin(yaw)
			face_y_map = robot_y + d[0] * math.sin(yaw) + d[1] * math.cos(yaw)
			face_coords_global = (face_x_map, face_y_map, float(d[2]))

			if face_id in self.closest_faces:
					stored_distance, _, _ = self.closest_faces[face_id]
					if current_dist >= stored_distance:
						continue 
			
			robot_coords_with_yaw = (robot_x, robot_y, yaw)
			self.closest_faces[face_id] = (current_dist, robot_coords_with_yaw, face_coords_3d)

			self._logger.info(f"Face 3D coordinates: {face_coords_3d}")
			self.get_logger().info(f"Robot position: x={self.robot_position[0]:.2f}, y={self.robot_position[1]:.2f}, Yaw: {self.robot_yaw:.2f}")
			self.save_dictionary_to_file(face_id, self.robot_position, face_coords_global, self.robot_yaw)
			self.save_whole_dictionary_to_file()



def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()