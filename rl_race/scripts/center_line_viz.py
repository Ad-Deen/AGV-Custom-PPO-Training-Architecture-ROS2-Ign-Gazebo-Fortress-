#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import numpy as np
from threading import Timer

class CenterLineVisualizer(Node):

    def __init__(self):
        super().__init__('center_line_visualizer')

        # Subscribe to the /center_points topic
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/center_points',
            self.center_line_callback,
            10
        )

        # Initialize the storage for all coordinates
        self.all_coordinates = []

        # Timer to save points after 5 seconds of inactivity
        self.save_timer = None
        self.timeout_duration = 5.0

        # Set up matplotlib for real-time plotting
        self.fig, self.ax = plt.subplots()
        self.scatter = self.ax.scatter([], [], s=50, c='b', marker='o')  # Larger dots
        self.ax.set_title('Center Line Visualization')
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.grid(True)

        # Set initial plot limits
        self.ax.set_xlim(-100, 100)  # Adjust these limits as needed
        self.ax.set_ylim(-150, 0)    # Adjust these limits as needed

        # Show the plot
        plt.ion()  # Turn on interactive mode
        plt.show()

    def center_line_callback(self, msg):
        # Extract coordinates from Float32MultiArray message
        data = msg.data
        if len(data) % 2 != 0:
            self.get_logger().warning('Received data length is not a multiple of 2, skipping.')
            return

        # Round coordinates to 1 decimal place and accumulate points
        new_coordinates = [(round(data[i], 1), round(data[i + 1], 1)) for i in range(0, len(data), 2)]
        self.all_coordinates.extend(new_coordinates)

        # Update the plot with accumulated coordinates
        self.update_plot()

        # Reset and start the timer for saving data
        self.reset_save_timer()

    def update_plot(self):
        if not self.all_coordinates:
            return

        # Separate the x and y coordinates for plotting
        x_coords, y_coords = zip(*self.all_coordinates)

        # Update scatter plot with accumulated x and y coordinates
        self.scatter.set_offsets(np.array(self.all_coordinates))

        # Update plot limits dynamically if needed
        self.ax.set_xlim(min(x_coords) - 10, max(x_coords) + 10)  # Adjust limits
        self.ax.set_ylim(min(y_coords) - 10, max(y_coords) + 10)  # Adjust limits

        # Redraw the plot
        self.fig.canvas.draw_idle()  # More efficient redrawing
        self.fig.canvas.flush_events()

    def reset_save_timer(self):
        # Cancel the existing timer if it exists
        if self.save_timer is not None:
            self.save_timer.cancel()

        # Start a new timer
        self.save_timer = Timer(self.timeout_duration, self.save_data)
        self.save_timer.start()

    def save_data(self):
        # Convert accumulated coordinates to a 2D NumPy array
        coordinates_array = np.array(self.all_coordinates)

        # Save the array to the specified file path
        save_path = '/home/deen/ros2_ws/src/rl_race/scripts/center_points.npy'
        np.save(save_path, coordinates_array)
        self.get_logger().info(f'Coordinates saved to {save_path}')

def main(args=None):
    rclpy.init(args=args)
    
    node = CenterLineVisualizer()

    # Run ROS2 spin loop
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
