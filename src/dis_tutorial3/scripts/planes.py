#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import pcl
import pcl_helper  # You'll need to implement this (see below)
import numpy as np
from std_msgs.msg import Header

class PlaneSegmenter(Node):
    def __init__(self):
        super().__init__('plane_segmenter')
        
        # Parameters
        self.declare_parameter('input_topic', '/oakd/rgb/preview/depth/points')
        self.input_topic = self.get_parameter('input_topic').value
        
        # Publisher and Subscriber
        self.publisher = self.create_publisher(PointCloud2, '/planes', 10)
        self.subscription = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.cloud_callback,
            10)
        
        self.get_logger().info(f"Listening to {self.input_topic}, publishing to /planes")

    def cloud_callback(self, msg):
        # Convert ROS PointCloud2 to PCL format
        cloud = pcl_helper.ros_to_pcl(msg)
        
        # Voxel Grid Downsampling
        vox = cloud.make_voxel_grid_filter()
        vox.set_leaf_size(0.01, 0.01, 0.01)
        cloud_filtered = vox.filter()
        
        # Plane Segmentation
        seg = cloud_filtered.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(0.01)
        seg.set_max_iterations(1000)
        
        remaining_indices = list(range(cloud_filtered.size()))
        planes_found = 0
        
        # Continue while >30% points remain
        while len(remaining_indices) > 0.3 * cloud_filtered.size():
            seg.set_indices(remaining_indices)
            inliers, coefficients = seg.segment()
            
            if len(inliers) == 0:
                break
            
            # Remove inliers from remaining indices
            remaining_indices = [i for i in remaining_indices if i not in inliers]
            planes_found += 1
        
        self.get_logger().info(f"Found {planes_found} planes")
        
        # Color non-planar points green
        cloud_out = pcl.PointCloud_PointXYZRGB()
        cloud_out.from_array(cloud_filtered.to_array())
        
        green_rgb = pcl_helper.rgb_to_float((0, 255, 0))
        for idx in remaining_indices:
            cloud_out[idx].rgb = green_rgb
        
        # Convert back to ROS and publish
        ros_cloud = pcl_helper.pcl_to_ros(cloud_out)
        self.publisher.publish(ros_cloud)

def main(args=None):
    rclpy.init(args=args)
    node = PlaneSegmenter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()