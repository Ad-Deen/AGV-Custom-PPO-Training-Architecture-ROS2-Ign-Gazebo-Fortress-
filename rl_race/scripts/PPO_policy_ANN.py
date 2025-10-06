#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool, Int32, Float32, String
from sensor_msgs.msg import LaserScan
import numpy as np
import tensorflow as tf
# from cv_bridge import CvBridge
# import matplotlib.pyplot as plt
import os
import subprocess
# from ranger import Ranger
from tensorflow.keras import regularizers  # type: ignore # Import regularizers


class PPOProcessor(Node):
    def __init__(self):
        super().__init__('ppo_processor')

        # Subscribe to the /scan topic for LaserScan data
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10
        )

        # Subscribe to the /action_taken topic
        self.action_taken_subscription = self.create_subscription(
            Int32,
            '/action_taken',
            self.action_taken_callback,
            10
        )

        # Subscribe to the /reward topic
        self.reward_subscription = self.create_subscription(
            Float32,
            '/reward',
            self.reward_callback,
            10
        )

        # Subscribe to the /episode_end topic to get Boolean triggers for the end of an episode
        self.episode_end_subscription = self.create_subscription(
            Bool,
            '/episode_end',
            self.episode_end_callback,
            10
        )

        # Publisher for action probabilities
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            '/action_space',
            10
        )

        self.sync_publisher = self.create_publisher(Bool, '/value_update_sync', 10)

        self.accum_penalty = 0  # Initialize accum_penalty counter

        # Initialize the ANN model for policy
        self.policy_model = self.create_ann_model()
        #Data  for PPO learning inspection over episodes
        self.episode_summary_pub = self.create_publisher(String, '/episode_summary', 10)

        # Check if the PPO_actor node is already running
        self.get_logger().info("Launching PPO_actor node...")
        self.run_ppo_actor_node()

        # Initialize lists to store episodic data
        self.states = []
        self.actions = []
        self.rewards = []
        # self.action_space = ['throttle', 'brake', 'left', 'right', 'throttle_left', 'throttle_right']

        # Initialize lists for reward accumulation
        self.scaled_rewards_accumulated = []
        self.scaled_rewards_per_episode = []

        # Initialize global min and max reward values
        self.global_min_reward = -20.00
        self.global_max_reward = 20.00

        #the discount of filtering bad decision based plays
        self.discount = 0.7

        self.current_cumulative_reward = 0

        # Initialize counters for policy updates and denials
        self.update_counter = 0  # Counts consecutive policy updates
        self.deny_counter = 0    # Counts consecutive policy denials

        # Publisher for intent (policy updates/denials)
        self.intent_pub = self.create_publisher(Int32, '/intent', 10)

        # Initialize episode tracker
        self.current_episode = 0
        self.previous_episode_reward = -np.inf  # To store the cumulative reward of the previous episode

        # Set up a timer for processing data and updating policy at 30 Hz
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        # To track whether the state, action, and reward should be processed
        self.process_state = False
        self.process_action = False
        self.process_reward = False
#...............................................................................................................................................................

    def run_ppo_actor_node(self):
        """
        Runs the PPO_actor.py ROS2 node using a bash command.
        """
        try:
            # Use subprocess to run the ROS2 node
            subprocess.Popen(["ros2", "run", "rl_race", "PPO_actor.py"])
            self.get_logger().info("PPO_actor node launched successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to launch PPO_actor node: {e}")

#...............................................................................................................................................................
    def create_ann_model(self):
        """
        Creates an optimized ANN model with dropout for regularization, batch normalization,
        and L2 regularization. If a saved model exists, loads the parameters.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(72,)),
            tf.keras.layers.Dense(256, activation='relu', 
                                kernel_regularizer=regularizers.l2(0.01),  # Apply L2 regularization
                                name='hidden_layer1'),
            tf.keras.layers.BatchNormalization(),  # Add batch normalization
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu', 
                                kernel_regularizer=regularizers.l2(0.01),  # Apply L2 regularization
                                name='hidden_layer2'),
            tf.keras.layers.BatchNormalization(),  # Add batch normalization
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu', 
                                kernel_regularizer=regularizers.l2(0.01),  # Apply L2 regularization
                                name='hidden_layer3'),
            tf.keras.layers.BatchNormalization(),  # Add batch normalization
            tf.keras.layers.Dense(32, activation='relu', 
                                kernel_regularizer=regularizers.l2(0.01),  # Apply L2 regularization
                                name='hidden_layer4'),
            tf.keras.layers.BatchNormalization(),  # Add batch normalization
            tf.keras.layers.Dense(6, activation='softmax')  # Output layer with no regularization
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Path to the saved model parameters
        saved_model_path = "/home/deen/ros2_ws/src/rl_race/scripts/NN_params.keras"

        # Check if the saved model exists
        if os.path.exists(saved_model_path):
            self.get_logger().info("Loading model parameters from saved file.")
            model = tf.keras.models.load_model(saved_model_path)
        else:
            self.get_logger().info("No saved model found, starting fresh.")

        return model
            
#............................................................................................................................................................

    def save_model(self):
        """
        Save the TensorFlow model to the specified path.
        """
        save_path = "/home/deen/ros2_ws/src/rl_race/scripts/NN_params.keras"
        self.policy_model.save(save_path)
        self.get_logger().info(f"Model saved to {save_path}")

#...............................................................................................................................................................
    def scale_rewards(self, rewards, gamma=0.95):
        """
        Calculates discounted rewards, applies a fixed range clipping, scales them between 0 and 1 based on 
        the actual range of discounted rewards, and deducts based on accum_penalty if consecutive poorer episodes occur.
        Penalization is applied only to the first half of the scaled rewards.
        """
        
        if len(rewards) == 0:
            self.get_logger().warning("Rewards list is empty, skipping reward scaling.")
            return []

        # Clip rewards to the range [-20, 20]
        rewards = np.clip(rewards, -20, 20)

        # Apply -15 penalty to the last reward for crashing
        rewards[-1] -= 15

        # Calculate discounted rewards
        discounted_rewards = np.zeros_like(rewards)
        cumulative_reward = 0
        for i in reversed(range(len(rewards))):
            cumulative_reward = rewards[i] + gamma * cumulative_reward
            discounted_rewards[i] = cumulative_reward

        # Calculate dynamic min and max for scaling
        min_discounted_reward = np.min(discounted_rewards)
        max_discounted_reward = np.max(discounted_rewards)
        range_reward = max_discounted_reward - min_discounted_reward

        # Avoid division by zero
        if range_reward == 0:
            scaled_rewards = np.zeros_like(discounted_rewards)
        else:
            # Scale rewards to [0, 1]
            scaled_rewards = (discounted_rewards - min_discounted_reward) / range_reward

        # Apply accum_penalty deduction to only the first half of the scaled rewards
        penalty_factor = 0.5 * self.accum_penalty
        half_len = len(scaled_rewards) // 2
        scaled_rewards[:half_len] = np.clip(scaled_rewards[:half_len] - penalty_factor, 0, 1)

        # Round the scaled rewards to 3 decimal places
        rounded_scaled_rewards = np.round(scaled_rewards, 3)

        return rounded_scaled_rewards



    
#...............................................................................................................................................................
    def listener_callback(self, msg):
        """
        Callback function for processing incoming laser scan data.
        """
        ranges = np.array(msg.ranges, dtype=np.float32)
        max_range = 75.0
        ranges[np.isinf(ranges)] = max_range
        ranges_normalized = ranges / max_range

        # Store the processed laser scan data for later use in the timer callback
        self.current_features = ranges_normalized

        # Set flag to indicate that new data is available
        self.process_state = True

#...............................................................................................................................................................
    def publish_action_probabilities(self, action_probs):
        """
        Publishes the action probabilities to the /action_space topic.
        """
        msg = Float32MultiArray()
        msg.data = action_probs.tolist()
        self.action_pub.publish(msg)

#...............................................................................................................................................................
    def action_taken_callback(self, msg):
        """
        Callback function for handling action taken messages.
        """
        action_index = int(msg.data)
        self.process_action = True
        self.current_action_index = action_index

#...............................................................................................................................................................
    def reward_callback(self, msg):
        """
        Callback function for handling reward messages.
        """
        reward_value = float(msg.data)
        self.process_reward = True
        self.current_reward = reward_value

#...............................................................................................................................................................
    def episode_end_callback(self, msg):
        """
        Callback function for handling episode end messages.
        """
        if msg.data:
            self.episode_end_handler()

#...............................................................................................................................................................
    def episode_end_handler(self):
        """
        This function should be called when an episode ends.
        """
        self.get_logger().info("episode end handler initiated")
        self.current_episode += 1
        scaled_rewards = self.scale_rewards(self.rewards)

        self.scaled_rewards_per_episode.append(scaled_rewards)
        self.rewards.clear()
        # self.get_logger().info(f"episode no. {self.current_episode}")
        
        self.update_policy()
            
        
        # Accumulate scaled rewards
        # self.scaled_rewards_accumulated.extend([item for sublist in self.scaled_rewards_per_episode for item in sublist])
        self.scaled_rewards_per_episode.clear()

         # Save model parameters every 10 episodes
        if self.current_episode % 5 == 0:
            self.save_model()
        # self.states.clear()
        # self.actions.clear()
        # self.rewards.clear()
        # Create and publish the episode summary
        episode_summary_msg = String()
        episode_summary_msg.data = f"Episode: {self.current_episode}, Cumulative Rewards: {self.current_cumulative_reward:.2f}"
        self.episode_summary_pub.publish(episode_summary_msg)

        # self.get_logger().info(f"Published episode summary: {episode_summary_msg.data}")


#...............................................................................................................................................................
    def timer_callback(self):
        """
        Timer callback for processing data, making predictions, and updating the policy at 30 Hz.
        """
        if self.process_state:
            # Ensure ANN input shape
            input_data = np.expand_dims(self.current_features, axis=0)

            # Predict action probabilities
            action_probs = self.policy_model.predict(input_data)

            # Publish the action probabilities
            self.publish_action_probabilities(action_probs[0])

            # Store the current state, action, and reward after publishing
            if self.process_action and self.process_reward:
                self.states.append(self.current_features)
                self.actions.append(self.current_action_index)
                self.rewards.append(self.current_reward)

                # Reset flags
                self.process_action = False
                self.process_reward = False

            # Reset the process_state flag to indicate prediction is complete
            self.process_state = False

#...............................................................................................................................................................
    def publish_value_update_sync(self, status):
        """
        Publishes a boolean status to the /value_update_sync topic.
        """
        msg = Bool()
        msg.data = status
        self.sync_publisher.publish(msg)

#..............................................................................................................................................................

    # def publish_intent(self):
    #     """
    #     Publishes the count of consecutive policy updates or denials to the /intent topic.
    #     """
    #     intent_msg = Int32()

    #     if self.update_counter > 0:
    #         intent_msg.data = self.update_counter  # Publish positive for updates
    #     else:
    #         intent_msg.data = -self.deny_counter  # Publish negative for denials

    #     self.get_logger().info(f"Publishing intent value: {intent_msg.data}")
    #     self.intent_pub.publish(intent_msg)

#...............................................................................................................................................................
    def update_policy(self):
        """
        Updates the policy using the collected data from the episode only if the cumulative reward 
        is greater than the previous episode. Accum_penalty tracks consecutive update skips.
        """
        self.scaled_rewards_accumulated.extend([item for sublist in self.scaled_rewards_per_episode for item in sublist])
        
        # Check if there's data to update the policy
        if len(self.states) == 0 or len(self.actions) == 0 or len(self.scaled_rewards_accumulated) == 0:
            self.get_logger().warning("No data available for policy update.")
            self.publish_value_update_sync(True)
            return

        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.scaled_rewards_accumulated)
        self.get_logger().info(f"Accumulated rewards : {self.scaled_rewards_accumulated}")

        # Calculate cumulative reward for the current episode
        self.current_cumulative_reward = np.sum(rewards)
        self.get_logger().info(f"Cumulative reward : {self.current_cumulative_reward}")

        # Compare the current episode's cumulative reward with the previous episode's
        if self.current_cumulative_reward > self.previous_episode_reward * self.discount or self.current_cumulative_reward < -50:
            # Perform policy update

            features = np.array(states)
            actions_one_hot = tf.keras.utils.to_categorical(actions, num_classes=6)

            if len(rewards) != len(features):
                self.get_logger().warning("Mismatch in lengths of scaled rewards and features.")
                self.publish_value_update_sync(False)
                return

            # Update the policy network
            self.policy_model.fit(features, actions_one_hot, sample_weight=rewards, epochs=1, verbose=1)

            # Update the previous episode's cumulative reward
            self.previous_episode_reward = self.current_cumulative_reward
            self.states.clear()
            self.actions.clear()
            self.scaled_rewards_accumulated.clear()

            # Reset accum_penalty on successful update
            self.accum_penalty = self.accum_penalty*0.5
            self.get_logger().info(f"Policy updated. Reset accum_penalty to {self.accum_penalty}")

            # Publish 'True' to indicate the policy update is successful
            self.publish_value_update_sync(True)
        else:
            # Policy update denied, increment the accum_penalty
            self.accum_penalty += 1
            self.get_logger().info(f"Skipping policy update. Incrementing accum_penalty to {self.accum_penalty}.")

            # Publish 'True' even if skipping to ensure other parts can proceed
            self.publish_value_update_sync(True)


#...............................................................................................................................................................
def main(args=None):
    rclpy.init(args=args)
    node = PPOProcessor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
