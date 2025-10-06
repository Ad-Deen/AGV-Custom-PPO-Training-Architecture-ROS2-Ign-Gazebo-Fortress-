#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import math
import subprocess
from time import time

class Organizer(Node):
    def __init__(self):
        super().__init__('organizer')
        
        # Subscribers
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        
        # Publisher for the "result" topic
        self.result_publisher = self.create_publisher(String, 'result', 10)
        
        # Initialize variables
        self.target_x = 145.31
        self.target_y = 34.684
        self.threshold = 5.0  # Threshold distance in meters
        self.collision_threshold = 2.0  # Collision detection threshold in meters
        self.robot_name = 'PPO_agent'  # Change this to your robot's name in Gazebo
        self.robot_pose = (0, 0)  # Initialize robot pose
        self.last_action_time = time()
        self.reset_count = 0  # Counter for number of resets

        # Timer to check for inactivity
        self.inactivity_timer = self.create_timer(1.0, self.check_inactivity)

    def odom_callback(self, msg):
        # Update robot pose
        self.robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)

        # Check distance to the target
        current_x, current_y = self.robot_pose
        distance = math.sqrt((current_x - self.target_x) ** 2 + (current_y - self.target_y) ** 2)

        if distance <= self.threshold:
            self.get_logger().info(f"Target reached. Publishing 'succeed!' and resetting robot position...")
            self.publish_result("succeed!")
            self.reset_robot_pose()
            self.last_action_time = time()

    def lidar_callback(self, msg):
        # Get the distances from the LiDAR scan
        distances = msg.ranges

        # Calculate the distance to track boundaries
        min_distance = min(distances)

        if min_distance < self.collision_threshold:
            self.get_logger().info(f"Collision detected. Publishing 'crashed' and resetting robot position...")
            self.publish_result("crashed")
            self.reset_robot_pose()
            self.last_action_time = time()

    def check_inactivity(self):
        # Check if 60 seconds have passed without any other event
        if time() - self.last_action_time >= 60.0:
            self.get_logger().info(f"Inactivity detected. Publishing 'lost' and resetting robot position...")
            self.publish_result("lost")
            self.reset_robot_pose()
            self.last_action_time = time()

    def publish_result(self, result):
        # Publish the result to the "result" topic
        msg = String()
        msg.data = f"{result}, Resets: {(self.reset_count-1)/3}"
        self.result_publisher.publish(msg)
        self.get_logger().info(f"Published result: {msg.data}")

    def reset_robot_pose(self):
        # Construct the Ignition command
        command = (
            'ign service -s /world/empty_world/set_pose '
            '--reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean '
            '--timeout 500 --req '
            '\'name: "PPO_agent", position: {x: 0.0, y: 0.0, z: 1.0}, '
            'orientation: {x: 0.0, y: 0.0, z: 0.707, w: 0.707}\''
        )
        
        self.get_logger().info(f'Executing Ignition command: {command}')

        # Run the command using subprocess
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.get_logger().info(f'Output:\n{result.stdout.decode()}')
            if result.stderr:
                self.get_logger().error(f'Error:\n{result.stderr.decode()}')
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f'Command failed with exit code {e.returncode}')
            self.get_logger().error(f'Error output: {e.stderr.decode()}')

        # Increment reset counter
        self.reset_count += 1
        self.get_logger().info(f"Total Resets: {(self.reset_count-1)/3}")

def main(args=None):
    rclpy.init(args=args)
    node = Organizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
