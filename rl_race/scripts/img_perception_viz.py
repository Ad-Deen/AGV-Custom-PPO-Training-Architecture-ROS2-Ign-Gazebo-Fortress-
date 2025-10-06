#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np
import re
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

class OccupancyGridCreator(Node):
    def __init__(self):
        super().__init__('occupancy_grid_creator')
        # Subscribe to the /img_perception topic
        self.subscription = self.create_subscription(
            String,
            '/img_perception',
            self.listener_callback,
            10
        )
        # Publisher for occupancy grid image
        self.grid_pub = self.create_publisher(
            Image,
            '/occupancy_grid_img',
            10
        )
        
        # Grid parameters
        self.grid_width = 72
        self.grid_height = 72
        self.grid = np.zeros((self.grid_height, self.grid_width))

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Initialize matplotlib plot for grid visualization
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.grid, cmap='gray', vmin=0, vmax=1)
        self.ax.set_xlim(0, self.grid_width)
        self.ax.set_ylim(self.grid_height, 0)
        self.ax.set_title('Occupancy Grid Visualization')
    
    def listener_callback(self, msg):
        data_str = msg.data
        coords = self.parse_coordinates(data_str)
        if coords:
            self.update_grid(coords)
            self.visualize_grid()
            self.publish_grid()

    def parse_coordinates(self, data_str):
        pattern = re.compile(r'\(([^,]+),([^\)]+)\)')
        matches = pattern.findall(data_str)
        coords = [(float(x), float(y)) for x, y in matches]
        return coords

    def update_grid(self, coords):
        self.grid.fill(0)
        x_min, x_max = 0, 90
        y_min, y_max = -30, 30
        for x, y in coords:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                x_idx = int((x - x_min) / (x_max - x_min) * (self.grid_width - 1))
                y_idx = int((y - y_min) / (y_max - y_min) * (self.grid_height - 1))
                self.grid[y_idx, x_idx] = 1

    def visualize_grid(self):
        self.im.set_data(self.grid)
        plt.draw()
        plt.pause(0.1)

    def publish_grid(self):
        # Convert grid to image message and publish
        grid_image = (self.grid * 255).astype(np.uint8)
        grid_msg = self.bridge.cv2_to_imgmsg(grid_image, encoding="mono8")
        self.grid_pub.publish(grid_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridCreator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
