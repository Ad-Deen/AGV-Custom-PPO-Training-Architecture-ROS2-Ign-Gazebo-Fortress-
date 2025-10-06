#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Int32, Float32
import numpy as np

class ActorNode(Node):

    def __init__(self):
        super().__init__('actor_node')

        # Initialize velocity and steering variables
        self.current_velocity = 0.0
        self.current_steering = 0.0

        # Subscriber to /action_space topic
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/action_space',
            self.action_space_callback,
            10)

        # Subscriber to /intent topic for exploration rate control
        # self.intent_subscription = self.create_subscription(
        #     Int32,
        #     '/intent',
        #     self.intent_callback,
        #     10)

        # Publisher for /cmd_vel topic
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Publisher for action index
        self.action_index_publisher = self.create_publisher(Int32, '/action_taken', 10)

        # Publisher for exploration rate
        # self.exploration_publisher = self.create_publisher(Float32, '/exploration', 10)

        # Timer for 30Hz refresh rate
        self.timer = self.create_timer(1/30.0, self.timer_callback)

        # Action parameters
        self.min_velocity = 2.0  # m/s
        self.max_velocity = 14.0  # m/s
        self.acceleration = 3.0  # m/s^2
        self.deceleration = 6.0  # m/s^2
        self.steering_limit = 0.6  # rad
        self.steering_speed = 1.0  # rad/s

        # Placeholder for the latest action probabilities
        self.latest_action_probs = None

        # Initial exploration rate (percentage of time to act randomly)
        self.exploration_rate = 0.90  # 50% exploration by default

        # Minimum and maximum exploration rates
        self.min_exploration_rate = 0.05  # Min exploration: 5%
        self.max_exploration_rate = 0.95  # Max exploration: 95%

    # Callback for intent message
    # def intent_callback(self, msg):
    #     intent_value = msg.data/100

    #     # Update the exploration rate based on intent:
    #     # More positive intent (policy updates) means lower exploration
    #     # Negative intent (policy denials) means higher exploration

    #     if intent_value > 0:
    #         # Reduce exploration as policy updates increase
    #         self.exploration_rate = max(
    #             self.min_exploration_rate,  # Limit to min exploration rate 0.05
    #             self.exploration_rate - abs(intent_value)
    #         )
    #     else:
    #         # Increase exploration as policy denials increase
    #         self.exploration_rate = min(
    #             self.max_exploration_rate,  # Limit to max exploration rate 0.95
    #             self.exploration_rate + abs(intent_value)
    #         )

    #     self.get_logger().info(f"Updated exploration rate: {self.exploration_rate:.2f}")

    def action_space_callback(self, msg):
        # Store the latest action probabilities
        action_probs = np.array(msg.data)
        
        # Normalize action probabilities
        if np.sum(action_probs) > 0:
            action_probs /= np.sum(action_probs)
        else:
            self.get_logger().warn("Sum of action probabilities is 0. Normalizing to uniform distribution.")
            action_probs = np.ones_like(action_probs) / len(action_probs)
        
        self.latest_action_probs = action_probs

    def timer_callback(self):
        if np.random.rand() < self.exploration_rate or self.latest_action_probs is None:
            # Choose a random action
            action_index = np.random.randint(0, 6)
            # self.get_logger().info("Exploring: Choosing random action")
        else:
            # Choose an action based on the given probabilities (exploitation)
            action_index = self.choose_action(self.latest_action_probs)
            # self.get_logger().info("Exploiting: Choosing best action")

        # Publish the chosen action index
        self.publish_action_index(action_index)

        # Compute the new velocity and steering based on the chosen action
        twist_msg = Twist()
        twist_msg.linear.x, twist_msg.angular.z = self.compute_command(action_index)
        
        # Publish the twist command
        self.cmd_vel_publisher.publish(twist_msg)

        # Publish the exploration rate as a percentage
        # self.publish_exploration_rate()

    # def publish_exploration_rate(self):
    #     # Publish the exploration rate as a Float32 message
    #     exploration_msg = Float32()
    #     exploration_msg.data = self.exploration_rate * 100  # Convert to percentage
    #     self.exploration_publisher.publish(exploration_msg)

    def choose_action(self, action_probs):
        """
        Choose the action with the highest probability, and randomly select among 
        actions with equal probabilities if there's a tie.
        """
        # Ensure action_probs is a 1D array
        action_probs = np.squeeze(action_probs)

        # Find the maximum probability value
        max_prob = np.max(action_probs)
        
        # Get all indices where the probability is equal to the maximum
        max_indices = np.where(action_probs == max_prob)[0]

        # Choose randomly among the indices with the highest probability
        action_index = np.random.choice(max_indices)

        # Optionally, print the index for debugging
        # self.get_logger().info(f"Chosen action index: {action_index}")

        return action_index

    def publish_action_index(self, action_index):
        """
        Publish the chosen action index as an Int32 message.
        """
        # Ensure action_index is of type int
        action_index = int(action_index)

        # Create and populate the Int32 message
        msg = Int32()
        msg.data = action_index

        # Publish the message
        self.action_index_publisher.publish(msg)

    def compute_command(self, action_index):
        # Define the available actions
        actions = ['throttle', 'brake', 'left', 'right', 'throttle_left', 'throttle_right']
        action = actions[action_index]

        # Compute the new velocity and steering based on the action
        if action == 'throttle':
            self.current_velocity = min(self.current_velocity + self.acceleration / 30.0, self.max_velocity)
        elif action == 'brake':
            self.current_velocity = max(self.current_velocity - self.deceleration / 30.0, self.min_velocity)
        elif action == 'throttle_left':
            self.current_velocity = min(self.current_velocity + self.acceleration / 30.0, self.max_velocity)
            self.current_steering = min(self.current_steering + self.steering_speed / 30.0, self.steering_limit)
        elif action == 'throttle_right':
            self.current_velocity = min(self.current_velocity + self.acceleration / 30.0, self.max_velocity)
            self.current_steering = max(self.current_steering - self.steering_speed / 30.0, -self.steering_limit)
        elif action == 'left':
            self.current_steering = min(self.current_steering + self.steering_speed / 30.0, self.steering_limit)
        elif action == 'right':
            self.current_steering = max(self.current_steering - self.steering_speed / 30.0, -self.steering_limit)

        # Ensure velocity is within the limits
        return np.clip(self.current_velocity, self.min_velocity, self.max_velocity), self.current_steering

def main(args=None):
    rclpy.init(args=args)
    node = ActorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
