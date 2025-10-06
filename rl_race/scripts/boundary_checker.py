import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

class BoundaryChecker(Node):
    def __init__(self):
        super().__init__('boundary_checker')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.robot_pose = None
        self.lidar_data = None

    def odom_callback(self, msg):
        # Extract the robot's position from the odometry message
        self.robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.plot_lidar_and_pose()

    def load_lidar_data(self, file_path, distance_threshold=0.01):
        try:
            # Load the .npy file
            data = np.load(file_path)
            if data.shape[0] == 2:
                self.lidar_data = data.T  # Transpose to make it Nx2
                print(f"Loaded data shape: {self.lidar_data.shape}")
                print(f"First few entries:\n{self.lidar_data[:5]}")

                # Downsample the LiDAR data
                self.lidar_data = self.downsample_points(self.lidar_data, distance_threshold)
                print(f"Downsampled data shape: {self.lidar_data.shape}")
            else:
                print("The .npy file is not in the expected format (Nx2).")
        except Exception as e:
            print(f"Failed to load the .npy file: {e}")

    def downsample_points(self, points, distance_threshold):
        # Use cKDTree to group points that are close together
        tree = cKDTree(points)
        indices = tree.query_ball_tree(tree, distance_threshold)

        # Retain only one point from each group of close points
        unique_indices = set()
        for group in indices:
            unique_indices.add(group[0])  # Take the first point in the group

        # Return the downsampled points
        return points[list(unique_indices)]

    def plot_lidar_and_pose(self):
        if self.lidar_data is None:
            return

        plt.figure(figsize=(10, 10))
        plt.scatter(self.lidar_data[:, 0], self.lidar_data[:, 1], s=0.01, color='black', label='LiDAR Points')

        if self.robot_pose:
            plt.scatter(self.robot_pose[0], self.robot_pose[1], color='red', label='Robot Pose', marker='o')
        
        plt.title('LiDAR Map with Robot Pose')
        plt.xlabel('X Coordinate (meters)')
        plt.ylabel('Y Coordinate (meters)')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()

        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = BoundaryChecker()
    
    # Path to the .npy file
    file_path = 'rl_race/scripts/lidar_points.npy'
    node.load_lidar_data(file_path, distance_threshold=0.01)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
