#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float32, Bool
import math

class RewardNode(Node):

    def __init__(self):
        super().__init__('reward_node')

        # Publisher to the /reward topic
        self.reward_publisher = self.create_publisher(Float32, '/reward', 10)

        # Subscriber to the /bot_perception topic (expecting Float64MultiArray)
        self.perception_subscription = self.create_subscription(
            Float64MultiArray,
            '/bot_perception',
            self.perception_callback,
            10
        )

        # Subscriber to /progress topic (Float32)
        self.progress_subscription = self.create_subscription(
            Float32,
            '/progress',
            self.progress_callback,
            10
        )

        self.track_trajectory = 0.0  # Placeholder for track trajectory
        self.velocity = 0.0  # Placeholder for velocity
        self.progress = 0.0  # Placeholder for track progression

        # Timer callback to run at 30Hz
        self.timer = self.create_timer(1/30.0, self.timer_callback)

        # Initialize operation time variables
        self.start_time = self.get_clock().now().to_msg().sec
        self.max_operation_time = 0.0
        self.current_operation_time = 0.0

        # Subscriber to /collisions topic (boolean)
        self.collision_subscription = self.create_subscription(
            Bool,
            '/collisions',
            self.collision_callback,
            10
        )

        self.collision_detected = False

    def perception_callback(self, msg):
        # Extract track trajectory (in degrees) and velocity (in m/s) from /bot_perception
        self.track_trajectory = msg.data[0]
        self.velocity = msg.data[1]

    def progress_callback(self, msg):
        # Extract progression data from /progress
        self.progress = msg.data

    def collision_callback(self, msg):
        self.collision_detected = msg.data

    def timer_callback(self):
        # Get the current operation time
        current_time = self.get_clock().now().to_msg().sec
        self.current_operation_time = current_time - self.start_time

        # Calculate reward/penalty based on track trajectory
        if abs(self.track_trajectory) > 0.2:        # Checks if the trajectory of the track relative to the bot moves more than 20%
            trajectory_reward = -5 * (abs(self.track_trajectory) - 0.19)   # Penalty scales with deviation
        else:
            trajectory_reward = 5 * (0.21 - abs(self.track_trajectory))    # Reward scales with alignment

        # Calculate the velocity reward
        if self.velocity < 30.0:
            velocity_reward = (-1 * (33.0 - self.velocity)) / 30
        else:
            velocity_reward = (1 * (self.velocity - 27.0)) / 14

        # Calculate the progression reward
        progression_reward = 10 * self.progress  # Scale progression reward

        # Handle collision case
        if self.collision_detected:
            # Penalize based on operation time before crashing
            penalty = 0
            total_reward = trajectory_reward + velocity_reward + progression_reward + penalty
            self.collision_detected = False  # Reset collision state
        else:
            total_reward = trajectory_reward + velocity_reward + progression_reward

        # Update the maximum operation time
        # self.max_operation_time = max(self.max_operation_time, self.current_operation_time)

        # Publish the total reward to the /reward topic
        reward_msg = Float32()
        reward_msg.data = total_reward
        self.reward_publisher.publish(reward_msg)

        # Log the rewards (optional)
        # self.get_logger().info(f"Track Trajectory: {self.track_trajectory} deg, Velocity: {self.velocity} m/s")
        # self.get_logger().info(f"Trajectory Reward: {trajectory_reward}, Velocity Reward: {velocity_reward}, Progression Reward: {progression_reward}, Total Reward: {total_reward}")

def main(args=None):
    rclpy.init(args=args)
    node = RewardNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
