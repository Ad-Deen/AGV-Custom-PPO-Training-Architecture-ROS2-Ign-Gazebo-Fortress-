#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
import numpy as np
import matplotlib.pyplot as plt

class ActionProbPlotter(Node):
    def __init__(self):
        super().__init__('action_prob_plotter')

        # Subscribe to the /action_space topic
        self.action_subscription = self.create_subscription(
            Float32MultiArray,
            '/action_space',
            self.action_space_callback,
            10
        )

        # Subscribe to the /exploration topic to get the exploration rate
        # self.exploration_subscription = self.create_subscription(
        #     Float32,
        #     '/exploration',
        #     self.exploration_callback,
        #     10
        # )

        self.action_space = ['throttle', 'brake', 'left', 'right', 'throttle_left', 'throttle_right']

        # Set up the initial plot
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.bars = self.ax.bar(self.action_space, [0]*len(self.action_space), color='blue', alpha=0.7)
        self.ax.set_xlabel('Actions')
        self.ax.set_ylabel('Probability')
        self.ax.set_ylim(0, 1)  # Ensure y-axis is scaled between 0 and 1

        # Initialize the exploration rate
        # self.exploration_rate = 0.0  # Default value

        # Set initial title
        self.ax.set_title('Action Probabilities | Exploration Rate: 50%')

        plt.ion()  # Turn on interactive mode
        plt.show()

    def action_space_callback(self, msg):
        """
        Callback function for handling action probabilities.
        """
        action_probs = np.array(msg.data)

        # Ensure action_probs is a 1D array
        action_probs = np.squeeze(action_probs)

        # Convert to list for plotting
        action_probs = action_probs.tolist()

        # Check for length consistency
        if len(self.action_space) != len(action_probs):
            self.get_logger().warning("Mismatch between action space length and action probabilities length")
            return

        # Update the plot
        for bar, prob in zip(self.bars, action_probs):
            bar.set_height(prob)

        # Update the title with the exploration rate and policy
        # self.ax.set_title(f'Action Probabilities | Exploration Rate: {self.exploration_rate:.2f}%')

        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # def exploration_callback(self, msg):
    #     """
    #     Callback function for handling the exploration rate.
    #     """
    #     # Update the exploration rate
    #     self.exploration_rate = msg.data

    #     # Update the title with the new exploration rate
    #     self.ax.set_title(f'Action Probabilities | Exploration Rate: {self.exploration_rate:.2f}%')

    #     # Redraw the figure
    #     self.fig.canvas.draw()
    #     self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    node = ActionProbPlotter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
