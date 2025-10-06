#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool, Int32, Float32
from sensor_msgs.msg import Image
import numpy as np
import tensorflow as tf
from cv_bridge import CvBridge
from sklearn.preprocessing import StandardScaler
import os

class PPOProcessor(Node):
    def __init__(self):
        super().__init__('ppo_processor')

        # Subscribe to the /occupancy_grid_img topic
        self.subscription = self.create_subscription(
            Image,
            '/occupancy_grid_img',
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

        # Publisher for action probabilities as Float64MultiArray
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            '/action_space',
            10
        )

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Initialize the CNN model for feature extraction
        self.cnn_model = self.create_cnn_model()

        # Initialize the Policy model for predicting action probabilities
        self.policy_model = self.create_policy_model()

        # **Added: Placeholder for old action probabilities for PPO**
        self.old_action_probs = [] 

        # Initialize lists to store episodic data
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []  # **Added: To store action probabilities**

        # Initialize global min and max reward values
        self.global_min_reward = -1.00
        self.global_max_reward = 1.00

        # Initialize episode tracker
        self.current_episode = 0

        # **Added: Path to save model weights**
        self.save_path = "/home/deen/ros2_ws/src/rl_race/scripts/policy_params.pth"

        # Set up a timer for processing data and updating policy at 30 Hz
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        # To track whether the state should be processed
        self.process_state = False
        self.process_action = False
        self.process_reward = False
#................................................................................................................................................................
    def create_cnn_model(self):
        """
        Creates the CNN model for processing the occupancy grid image.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(72, 72, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu')  # Feature extraction layer
        ])
        return model
#................................................................................................................................................................
    def create_policy_model(self):
        """
        Creates the Policy model for predicting action probabilities.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(256,)),  # Assuming output from CNN is 256-dimensional
            tf.keras.layers.Dense(6, activation='softmax')  # Output layer for 6 possible actions
        ])
        return model
#................................................................................................................................................................
    def ppo_loss(self, old_probs, new_probs, advantages, epsilon=0.2, entropy_beta=0.01):
        """
        Custom PPO loss function with entropy regularization.
        """
        prob_ratio = tf.exp(new_probs - old_probs)
        clipped_ratio = tf.clip_by_value(prob_ratio, 1 - epsilon, 1 + epsilon)
        surrogate = tf.minimum(prob_ratio * advantages, clipped_ratio * advantages)

        # Entropy regularization
        entropy = -tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-10), axis=-1)
        entropy_loss = entropy_beta * entropy

        return -tf.reduce_mean(surrogate + entropy_loss)
#................................................................................................................................................................
    def scale_rewards(self, rewards, gamma=0.99):
        """
        Calculates discounted rewards and scales them between 0 and 1 
        based on the absolute value of global min and max rewards.

        Parameters:
        - rewards (np.array): Array of reward values.
        - gamma (float): Discount factor (default is 0.99).

        Returns:
        - np.array: Scaled rewards between 0 and 1.
        """
        # Apply -15 penalty to the last reward
        rewards[-1] -= 10
        rewards = np.clip(rewards, -10, 10)  # Clip extreme rewards
        # Step 1: Calculate discounted rewards
        discounted_rewards = np.zeros_like(rewards)
        cumulative_reward = 0
        
        # Compute discounted rewards from last to first
        for i in reversed(range(len(rewards))):
            cumulative_reward = rewards[i] + gamma * cumulative_reward
            discounted_rewards[i] = cumulative_reward
        
        # Step 2: Update global min and max rewards based on absolute values
        current_min_reward = np.min(discounted_rewards)
        current_max_reward = np.max(discounted_rewards)

        # Find the maximum absolute value between the current rewards
        max_abs_reward = max(abs(current_min_reward), abs(current_max_reward))

        # Update the global min and max based on the absolute maximum
        if abs(current_min_reward) > abs(self.global_min_reward):
            self.global_min_reward = -max_abs_reward
        if abs(current_max_reward) > abs(self.global_max_reward):
            self.global_max_reward = max_abs_reward

        # Step 3: Use absolute global min and max rewards for scaling
        range_reward = self.global_max_reward - self.global_min_reward

        self.get_logger().info(f"Discounted rewards: {discounted_rewards}")
        
        # Prevent division by zero if all rewards are the same
        if range_reward == 0:
            scaled_rewards = np.zeros_like(discounted_rewards)
        else:
            scaled_rewards = (discounted_rewards - self.global_min_reward) / range_reward

        return scaled_rewards


#................................................................................................................................................................
    def listener_callback(self, msg):
        """
        Callback function for processing incoming LiDAR occupancy grid images.
        """
        # Convert ROS image to OpenCV image
        grid_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        self.get_logger().info(f"incoming grid {grid_image}")
        grid_image = grid_image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        self.get_logger().info(f"reshaped grid {grid_image}")

        # Reshape grid for CNN (add batch dimension and channel dimension)
        grid_reshaped = np.expand_dims(np.expand_dims(grid_image, axis=-1), axis=0)

        # Use CNN to extract features
        features = self.cnn_model.predict(grid_reshaped)

        # Predict action probabilities using the policy model
        action_probs = self.policy_model.predict(features)
        # self.get_logger().info(f"Action probs: {action_probs}")

        # **Update: Store old action probabilities for PPO**
        self.old_action_probs = action_probs[0]  # Storing old probabilities

        # Set flag to process state and action
        self.process_state = True
        self.current_features = features[0]
        self.publish_action_probabilities(action_probs[0])
#................................................................................................................................................................
    def publish_action_probabilities(self, action_probs):
        """
        Publishes the action probabilities to the /action_space topic.

        Parameters:
        - action_probs (np.array): Array of action probabilities.
        """
        msg = Float32MultiArray()
        msg.data = action_probs.tolist()
        self.action_pub.publish(msg)
#................................................................................................................................................................
    def action_taken_callback(self, msg):
        """
        Callback function for handling action taken messages.
        """
        # Extract the action index from the message
        action_index = int(msg.data)  # Assuming single value for action index
        self.process_action = True
        self.current_action_index = action_index
#................................................................................................................................................................
    def reward_callback(self, msg):
        """
        Callback function for handling reward messages.
        """
        # Extract the reward value from the message
        reward_value = float(msg.data)  # Assuming single value for reward
        self.process_reward = True
        self.current_reward = reward_value
#................................................................................................................................................................
    def episode_end_callback(self, msg):
        """
        Callback function for handling episode end messages.
        """
        if msg.data:
            self.episode_end_handler()
#................................................................................................................................................................
    def episode_end_handler(self):
        """
        This function should be called when an episode ends.
        """
        # Increment episode counter
        self.current_episode += 1

        # Perform policy update
        self.update_policy()

        # Reset states, actions, rewards for the new episode
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.action_probs.clear()
        self.old_action_probs = []  # Reset old action probabilities

        # Save model weights every 20 episodes
        if self.current_episode % 20 == 0:
            self.save_model_weights()
#................................................................................................................................................................
    def save_model_weights(self):
        """
        Saves the policy model's weights to a file.
        """
        self.policy_model.save_weights(self.save_path)
        self.get_logger().info("NN params saved")  # Log message when parameters are saved
#................................................................................................................................................................
    def timer_callback(self):
        """
        Timer callback for processing data and updating policy at 30 Hz.
        """
        if self.process_state and self.process_action and self.process_reward:
            # Store state, action, reward, and action probability
            self.states.append(self.current_features)
            self.actions.append(self.current_action_index)
            self.rewards.append(self.current_reward)
            self.action_probs.append(self.old_action_probs)  # **Store old action probabilities**

            # Reset flags
            self.process_state = False
            self.process_action = False
            self.process_reward = False
#................................................................................................................................................................
    def update_policy(self):
        """
        Updates the policy using the collected data from the episode.
        """
        if len(self.states) == 0 or len(self.actions) == 0 or len(self.rewards) == 0:
            self.get_logger().warning("No data available for policy update.")
            return
        
        # Convert lists to numpy arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        old_action_probs = np.array(self.action_probs)  # **Convert old action probs to numpy array**

        # Calculate advantages (for simplicity, using scaled rewards as advantages)
        scaled_rewards = self.scale_rewards(rewards)
        advantages = scaled_rewards - np.mean(scaled_rewards)
        advantages /= np.std(advantages)

        # One-hot encode actions
        actions_one_hot = tf.keras.utils.to_categorical(actions, num_classes=6)

        # Train the policy model using the custom PPO loss
        with tf.GradientTape() as tape:
            # Predict new probabilities
            new_probs = self.policy_model(states, training=True)
            # Calculate custom PPO loss
            loss = self.ppo_loss(old_action_probs, new_probs, advantages)
            self.get_logger().info(f"Loss: {loss}")

        # Calculate gradients and apply them
        grads = tape.gradient(loss, self.policy_model.trainable_variables)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables))
#................................................................................................................................................................
def main(args=None):
    rclpy.init(args=args)
    node = PPOProcessor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
