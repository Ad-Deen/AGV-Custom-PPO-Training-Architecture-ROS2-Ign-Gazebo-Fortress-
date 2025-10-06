#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
import numpy as np

class LiDARRangeMapNode(Node):
    def __init__(self):
        super().__init__('lidar_range_map_node')

        # Subscription to the /scan topic to get LiDAR data
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publisher for the perception grid data on /perception_grid topic
        self.perception_grid_publisher = self.create_publisher(
            Float32MultiArray,
            '/perception_grid',
            10
        )

        # Define range map parameters
        self.map_width = 72  # Number of bins for width (corresponding to angle resolution)
        self.map_height = 72  # Number of bins for distance
        self.max_range = 75.0  # Maximum range of the LiDAR in meters

    def scan_callback(self, msg):
        # Convert LaserScan ranges to numpy array
        ranges = np.array(msg.ranges)
        
        # Clip ranges to max range and normalize
        ranges = np.clip(ranges, 0, self.max_range) / self.max_range
        
        # Convert 1D ranges to 2D range map
        range_map = self.convert_to_range_map(ranges)
        
        # Flatten the range map for publishing
        range_map_flat = range_map.flatten()

        # Create a Float32MultiArray message
        perception_grid_msg = Float32MultiArray()
        perception_grid_msg.data = range_map_flat.tolist()

        # Publish the perception grid
        self.perception_grid_publisher.publish(perception_grid_msg)
        # self.get_logger().info('Published perception grid to /perception_grid')

    def convert_to_range_map(self, ranges):
        # Create an empty 2D grid
        range_map = np.zeros((self.map_height, self.map_width))

        # Calculate angle increment based on -55 to 55 degrees
        angle_increment = 110 / self.map_width  # Total coverage is 110 degrees

        # Center of the grid represents the robot's position
        center_x, center_y = self.map_width // 2, self.map_height // 2

        # Mark the robot's position in the center of the grid
        range_map[center_y, center_x] = -1  # Use -1 or another marker to indicate the robot's position

        for i, r in enumerate(ranges):
            # Calculate corresponding bin for the range
            range_bin = int(r * (self.map_height - 1))

            # Calculate corresponding bin for the angle
            angle_bin = i

            # Set the cell in the range map
            range_map[range_bin, angle_bin] = 1  # Mark as an obstacle presence

        return range_map

def main(args=None):
    rclpy.init(args=args)
    node = LiDARRangeMapNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
