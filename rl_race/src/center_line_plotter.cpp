#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>

// Function to calculate Euclidean distance between two 2D points
double calculateDistance2D(const std::pair<double, double>& p1, const std::pair<double, double>& p2) {
    return std::sqrt(std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2));
}

// Check if two points are approximately equal within a tolerance (2D version)
bool arePointsEqual2D(const std::pair<double, double>& p1, const std::pair<double, double>& p2, double tolerance) {
    return calculateDistance2D(p1, p2) < tolerance;
}

class OdomSubscriber : public rclcpp::Node {
public:
    OdomSubscriber()
    : Node("odom_subscriber"), 
      threshold_(2.0), 
      return_threshold_(3.0), 
      initial_pose_recorded_(false), 
      gathering_coordinates_(true), 
      has_moved_away_(false),
      last_received_time_(std::chrono::steady_clock::now()) {
        // Subscriber to the /odom topic
        subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&OdomSubscriber::odomCallback, this, std::placeholders::_1));
        
        // Publisher to the /center_points topic
        publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/center_points", 10);
    }

    void checkForTimeout() {
        auto current_time = std::chrono::steady_clock::now();
        auto duration_since_last_point = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_received_time_).count();
        
        if (duration_since_last_point > 5) {
            // Save the accumulated points to a file
            std::ofstream file("/home/deen/ros2_ws/src/rl_race/scripts/center_points.txt");
            if (file.is_open()) {
                for (const auto& point : stored_positions_) {
                    file << point.first << " " << point.second << "\n";
                }
                file.close();
                RCLCPP_INFO(this->get_logger(), "Saved center points to /home/deen/ros2_ws/src/rl_race/scripts/center_points.txt");
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to open file for saving center points.");
            }
            gathering_coordinates_ = false; // Stop gathering coordinates
        }
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        if (!gathering_coordinates_) {
            return;  // Stop gathering if we have finished collecting
        }

        // Extract current position from the Odometry message
        std::pair<double, double> current_position = {
            msg->pose.pose.position.x,
            msg->pose.pose.position.y
        };

        // Record the initial position
        if (!initial_pose_recorded_) {
            initial_pose_ = current_position;
            stored_positions_.push_back(current_position);
            initial_pose_recorded_ = true;
            last_received_time_ = std::chrono::steady_clock::now();  // Reset the timer
            return;
        }

        // Check if we have moved away from the initial position
        if (!has_moved_away_) {
            if (calculateDistance2D(initial_pose_, current_position) > return_threshold_) {
                has_moved_away_ = true;
                RCLCPP_INFO(this->get_logger(), "Bot has moved away from the initial position.");
            }
            return; // Continue to check until the bot has moved away
        }

        // Check if we have returned close to the initial position
        if (arePointsEqual2D(initial_pose_, current_position, return_threshold_)) {
            RCLCPP_INFO(this->get_logger(), "Returned close to initial position within %.2f meters, stopping coordinate gathering.", return_threshold_);
            gathering_coordinates_ = false;
            return;
        }

        // Store and publish new positions if distance is greater than the threshold
        if (!stored_positions_.empty()) {
            double distance = calculateDistance2D(stored_positions_.back(), current_position);
            if (distance >= threshold_) {
                stored_positions_.push_back(current_position);
                last_received_time_ = std::chrono::steady_clock::now();  // Update the last received time
                publishCoordinate(current_position);  // Publish the new coordinate immediately
            }
        }
    }

    void publishCoordinate(const std::pair<double, double>& point) {
        auto float_array_msg = std::make_shared<std_msgs::msg::Float32MultiArray>();
        float_array_msg->data = { static_cast<float>(point.first), static_cast<float>(point.second) };

        // Define the array dimensions
        float_array_msg->layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
        float_array_msg->layout.dim[0].label = "coordinates";
        float_array_msg->layout.dim[0].size = 2;
        float_array_msg->layout.dim[0].stride = 2;

        publisher_->publish(*float_array_msg);
        RCLCPP_INFO(this->get_logger(), "Published position: (%.2f, %.2f) to /center_points", point.first, point.second);
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr publisher_;
    std::vector<std::pair<double, double>> stored_positions_;  // Vector to store the positions
    std::pair<double, double> initial_pose_;  // To store the initial position
    double threshold_;  // Distance threshold for storing points
    double return_threshold_;  // Distance threshold for returning to the initial position
    bool initial_pose_recorded_;  // Flag to check if the initial pose has been recorded
    bool gathering_coordinates_;  // Flag to control coordinate gathering
    bool has_moved_away_;  // Flag to check if the bot has moved away from the initial position
    std::chrono::steady_clock::time_point last_received_time_;  // To keep track of the last received time
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OdomSubscriber>();

    // Main loop
    rclcpp::Rate rate(10);  // 10 Hz
    while (rclcpp::ok()) {
        rclcpp::spin_some(node);
        node->checkForTimeout();  // Check for timeout every loop
        rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
