#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32, Float32
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
import numpy as np
from cv_bridge import CvBridge
import os

class PPODataCollector(Node):
    def __init__(self):
        super().__init__('ppo_data_collector')

        # Subscribers
        self.create_subscription(Image, '/occupancy_grid_img', self.occupancy_grid_callback, 10)
        self.create_subscription(Int32, '/action_taken', self.action_taken_callback, 10)
        self.create_subscription(Float32, '/reward', self.reward_callback, 10)
        self.create_subscription(Float64MultiArray, '/action_space', self.action_space_callback, 10)
        self.create_subscription(Int32, '/system_feedback', self.system_feedback_callback, 10)
        self.create_subscription(Float32, '/value_est', self.value_est_callback, 10)
        self.create_subscription(Float32, '/discount_factor', self.discount_factor_callback, 10)
        self.create_subscription(Float64MultiArray, '/bot_perception', self.bot_perception_callback, 10)
        self.create_subscription(Bool, '/episode_end', self.episode_end_callback, 10)  # New subscription

        # Initialize CV Bridge for image conversion
        self.bridge = CvBridge()

        # Episode data storage using numpy arrays
        self.episode_data = []
        self.current_episode_number = None

        # Temporary storage for the current step data
        self.current_state = None
        self.next_state = None
        self.action_taken = None
        self.reward = None
        self.action_probs = None
        self.value_est = None
        self.discount_factor = 0.8  # Default discount factor initialization
        self.bot_perception_data = None  # Placeholder for bot perception data
        self.step_count = 0

        # Path to save the episodic data (use a fixed file name)
        self.save_path = '/home/deen/ros2_ws/src/rl_race/trajectories/'
        self.file_name = 'latest_episode_data.npy'  # Change to .npy file format

        # Timer to ensure node runs at 30Hz
        self.create_timer(1.0 / 30.0, self.timer_callback)

    def occupancy_grid_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        grid_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

        # Update states
        if self.current_state is None:
            self.current_state = grid_image
        else:
            self.next_state = grid_image
        # self.get_logger().info(f"Grid {self.next_state}")

    def action_taken_callback(self, msg):
        # Assuming action_taken is published as an integer of the chosen action index
        self.action_taken = int(msg.data)
        self.get_logger().info(f"action taken {self.action_taken}")

    def reward_callback(self, msg):
        # Get the reward value
        self.reward = msg.data
        self.get_logger().info(f"Reward = {self.reward}")

    def action_space_callback(self, msg):
        # Get the action probabilities
        self.action_probs = np.array(msg.data)
        # self.get_logger().info(f"action taken {self.action_probs}")

    def value_est_callback(self, msg):
        # Get the value estimation
        self.value_est = msg.data
        # self.get_logger().info(f"Value {self.value_est}")

    def discount_factor_callback(self, msg):
        # Update the discount factor dynamically from the Value NN node
        self.discount_factor = msg.data
        # self.get_logger().info(f"action taken {self.action_taken}")

    def bot_perception_callback(self, msg):
        # Store bot perception data directly from the topic
        self.bot_perception_data = np.array(msg.data)

        # Normalize bot_perception_data[0] by 1 and bot_perception_data[1] by 47
        if len(self.bot_perception_data) > 1:
            self.bot_perception_data[0] = self.bot_perception_data[0] / 1
            self.bot_perception_data[1] = self.bot_perception_data[1] / 47
        # self.get_logger().info(f"action taken {self.action_taken}")

    def system_feedback_callback(self, msg):
        # Handle new episode number from /system_feedback
        episode_number = msg.data

        # If this is a new episode, save the previous one and reset data
        if episode_number != self.current_episode_number:
            if self.episode_data:
                self.save_episode_data()

            self.current_episode_number = episode_number
            self.step_count = 0  # Reset step count for the new episode

    def episode_end_callback(self, msg):
        if msg.data:
            self.save_episode_data()

    def save_step_data(self):
        # Ensure all required data are available before saving
        if (self.current_state is not None and self.next_state is not None and 
            self.action_taken is not None and self.reward is not None and 
            self.action_probs is not None and self.value_est is not None and
            self.bot_perception_data is not None):
            
            # Calculate log probability of the chosen action
            action_log_prob = np.log(self.action_probs[self.action_taken])

            # Append the step data as a numpy array
            step_data = np.array([
                self.current_state,        # Current state (Occupancy grid image)
                self.next_state,           # Next state (Occupancy grid image)
                self.action_taken,         # Action taken (int)
                self.reward,               # Reward (float)
                self.action_probs,         # Action probabilities (numpy array)
                action_log_prob,           # Log probability of the action (float)
                self.value_est,            # Value estimation (float)
                self.discount_factor,      # Discount factor (float)
                self.bot_perception_data   # Bot perception data (numpy array)
            ], dtype=object)  # Use dtype=object for mixed data types

            # Append the step data to the episode data
            self.episode_data.append(step_data)
            self.step_count += 1

            # Log detailed step data for debugging
            # self.get_logger().info(f"Step {self.step_count}:")
            # self.get_logger().info(f"  Current State: {self.current_state}")
            # self.get_logger().info(f"  Next State: {self.next_state}")
            # self.get_logger().info(f"  Action Taken: {self.action_taken}")
            # self.get_logger().info(f"  Reward: {self.reward}")
            # self.get_logger().info(f"  Action Probabilities: {self.action_probs}")
            # self.get_logger().info(f"  Action Log Probability: {action_log_prob}")
            # self.get_logger().info(f"  Value Estimation: {self.value_est}")
            # self.get_logger().info(f"  Discount Factor: {self.discount_factor}")
            # self.get_logger().info(f"  Bot Perception Data: {self.bot_perception_data}")

            # Reset step data
            self.current_state = self.next_state
            self.next_state = None
            self.action_taken = None
            self.reward = None
            self.action_probs = None
            self.value_est = None
            self.bot_perception_data = None

    def print_episode_data(self):
        # Full path to the .npy file
        file_path = os.path.join(self.save_path, self.file_name)

        # Load the data from file
        episode_data = np.load(file_path, allow_pickle=True)

        # Log the dataset to the console
        # self.get_logger().info("Episode Data:")
        # for idx, step in enumerate(episode_data):
        #     self.get_logger().info(f"Step {idx + 1}:")
        #     self.get_logger().info(f"  Current State: {step[0]}")
        #     self.get_logger().info(f"  Next State: {step[1]}")
        #     self.get_logger().info(f"  Action Taken: {step[2]}")
        #     self.get_logger().info(f"  Reward: {step[3]}")
        #     self.get_logger().info(f"  Action Probabilities: {step[4]}")
        #     self.get_logger().info(f"  Action Log Probability: {step[5]}")
        #     self.get_logger().info(f"  Value Estimation: {step[6]}")
        #     self.get_logger().info(f"  Discount Factor: {step[7]}")
        #     self.get_logger().info(f"  Bot Perception Data: {step[8]}")
        #     self.get_logger().info("\n")

    def save_episode_data(self):
        # Full path to the .npy file (constant file name to overwrite)
        file_path = os.path.join(self.save_path, self.file_name)

        # Convert episode data to numpy array and save to .npy file
        np.save(file_path, np.array(self.episode_data, dtype=object))

        # Log message to console
        # self.get_logger().info(f"Episode {self.current_episode_number} data saved to {file_path}.")

        # Print the episode data
        self.print_episode_data()

        # Clear the episode data after saving
        self.episode_data = []

    def timer_callback(self):
        # Save step data if all required data are available
        if (self.current_state is not None and self.next_state is not None and 
            self.action_taken is not None and self.reward is not None and 
            self.action_probs is not None and self.value_est is not None and
            self.bot_perception_data is not None):
            
            self.save_step_data()

def main(args=None):
    rclpy.init(args=args)
    node = PPODataCollector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
