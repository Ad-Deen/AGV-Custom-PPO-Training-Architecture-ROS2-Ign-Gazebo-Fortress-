#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

class EpisodePlotter(Node):
    def __init__(self):
        super().__init__('episode_plotter')

        # Initialize lists to store episode data
        self.episode_numbers = []
        self.cumulative_rewards = []

        # Create a subscription to the /episode_summary topic
        self.subscription = self.create_subscription(
            String,
            '/episode_summary',
            self.episode_summary_callback,
            10
        )

        # Initialize the plot
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-', marker='o', label='Cumulative Rewards')
        self.ax.set_xlabel('Episode Number')
        self.ax.set_ylabel('Cumulative Rewards')
        self.ax.set_title('Episode vs Cumulative Rewards')
        self.ax.legend()
        plt.ion()  # Interactive mode on
        plt.show()

    def episode_summary_callback(self, msg):
        """
        Callback function to handle incoming episode summary messages.
        """
        # Parse the message
        data = msg.data
        try:
            episode_str, rewards_str = data.split(', ')
            episode_number = int(episode_str.split(': ')[1])
            cumulative_reward = float(rewards_str.split(': ')[1])
            
            # Update data lists
            self.episode_numbers.append(episode_number)
            self.cumulative_rewards.append(cumulative_reward)

            # Update the plot
            self.update_plot()
        except Exception as e:
            self.get_logger().error(f"Failed to parse message: {data}, Error: {e}")

    def update_plot(self):
        """
        Update the plot with new data.
        """
        self.line.set_data(self.episode_numbers, self.cumulative_rewards)
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Redraw the plot
        plt.draw()
        plt.pause(0.01)

def main(args=None):
    rclpy.init(args=args)
    node = EpisodePlotter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
