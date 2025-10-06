#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import torch
import torch.nn.functional as F
from std_msgs.msg import Float64MultiArray, Float32

class PPOLossNode(Node):
    def __init__(self):
        super().__init__('ppo_loss_calculator')

        # Subscribe to the topics for action probabilities and value estimates
        self.create_subscription(Float64MultiArray, '/action_space', self.action_probs_callback, 10)
        self.create_subscription(Float32, '/est_value', self.value_estimate_callback, 10)
        self.create_subscription(Float32, '/reward', self.reward_callback, 10)

        # Publisher for policy loss to the policy update node
        self.policy_loss_publisher = self.create_publisher(Float32, '/policy_loss', 10)

        # Variables to store data
        self.old_action_probs = None
        self.new_action_probs = None
        self.value_estimates = None
        self.reward = None
        self.gamma = 0.99  # Discount factor
        self.eps_clip = 0.2  # PPO clip parameter

        # Buffer for the first action probabilities
        self.buffered_action_probs = None

    def action_probs_callback(self, msg: Float64MultiArray):
        # Convert message data to tensor
        data = torch.tensor(msg.data, dtype=torch.float32)

        if self.buffered_action_probs is None:
            # Buffer the first action probabilities
            self.buffered_action_probs = data
            self.get_logger().info("Buffered first action probabilities.")
        else:
            # Update old_action_probs and new_action_probs
            self.old_action_probs = self.buffered_action_probs
            self.new_action_probs = data

            # Update the buffered action probabilities for the next step
            self.buffered_action_probs = data

            # Log the action probabilities
            # self.get_logger().info(f'Old Action Probabilities: {self.old_action_probs.numpy()}')
            # self.get_logger().info(f'New Action Probabilities: {self.new_action_probs.numpy()}')

    def value_estimate_callback(self, msg: Float32):
        self.value_estimates = torch.tensor([msg.data], dtype=torch.float32)

    def reward_callback(self, msg: Float32):
        self.reward = torch.tensor([msg.data], dtype=torch.float32)

    def compute_and_publish_policy_loss(self):
        if (self.old_action_probs is None or self.new_action_probs is None or 
            self.reward is None or self.value_estimates is None):
            # self.get_logger().info("Waiting for all required data to compute loss...")
            return

        # Assuming value_estimates are V(s_t) and reward is r_t
        value_s_t = self.value_estimates  # V(s_t)
        value_s_t_plus_1 = self.reward + self.gamma * value_s_t  # V(s_{t+1}) (target)

        # Advantage: A(s_t) = r_t + gamma * V(s_{t+1}) - V(s_t)
        advantage = value_s_t_plus_1 - value_s_t

        # Ratio of new to old action probabilities
        prob_ratio = torch.exp(self.new_action_probs - self.old_action_probs)

        # Clipped surrogate loss function
        clipped_prob_ratio = torch.clamp(prob_ratio, 1 - self.eps_clip, 1 + self.eps_clip)

        # PPO loss: taking the minimum of the clipped and non-clipped
        loss = -torch.min(prob_ratio * advantage, clipped_prob_ratio * advantage).mean()

        # Log the calculated loss and advantage
        # self.get_logger().info(f"Advantage: {advantage.item()}, PPO Loss: {loss.item()}")

        # Publish the policy loss for the policy node to use
        loss_msg = Float32()
        loss_msg.data = loss.item()
        self.policy_loss_publisher.publish(loss_msg)

        # Reset the stored data for the next update cycle
        self.old_action_probs = None
        self.new_action_probs = None
        self.value_estimates = None
        self.reward = None

def main(args=None):
    rclpy.init(args=args)

    ppo_loss_node = PPOLossNode()

    # Run the node at a fixed rate (e.g., 30Hz)
    ppo_loss_node.create_timer(1.0 / 30, ppo_loss_node.compute_and_publish_policy_loss)

    rclpy.spin(ppo_loss_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
