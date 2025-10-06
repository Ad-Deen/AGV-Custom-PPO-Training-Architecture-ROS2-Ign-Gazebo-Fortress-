#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/float32.hpp"  // Include header for Float32 messages
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <utility>

// Function to calculate Euclidean distance between two 2D points
double calculateDistance2D(const std::pair<double, double>& p1, const std::pair<double, double>& p2) {
    return std::sqrt(std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2));
}

// Function to read center points from a file
std::vector<std::pair<double, double>> loadCenterPoints(const std::string& filepath) {
    std::vector<std::pair<double, double>> points;
    std::ifstream file(filepath);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            double x, y;
            if (iss >> x >> y) {
                points.emplace_back(x, y);
            }
        }
        file.close();
    } else {
        throw std::runtime_error("Failed to open file for reading center points.");
    }
    return points;
}

class TrackProgressionNode : public rclcpp::Node {
public:
    TrackProgressionNode()
    : Node("track_progression_node"),
      center_points_path_("/home/deen/ros2_ws/src/rl_race/scripts/center_points.txt"),
      center_points_(loadCenterPoints(center_points_path_)) {
        // Subscriber to the /odom topic
        subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&TrackProgressionNode::odomCallback, this, std::placeholders::_1));

        // Publisher for the /progress topic
        progress_publisher_ = this->create_publisher<std_msgs::msg::Float32>("/progress", 10);
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        // Extract bot's current position
        std::pair<double, double> bot_position = {
            msg->pose.pose.position.x,
            msg->pose.pose.position.y
        };

        // Calculate the bot's progression along the track
        double progression = calculateProgression(bot_position);

        // Log the bot's position and track progression
        // RCLCPP_INFO(this->get_logger(), "Bot position: (%.2f, %.2f), Track progression: %.2f%%", 
        //             bot_position.first, bot_position.second, progression * 100);

        // Publish the progression data to /progress topic
        auto msg_progress = std_msgs::msg::Float32();
        msg_progress.data = static_cast<float>(progression);
        progress_publisher_->publish(msg_progress);
    }

    double calculateProgression(const std::pair<double, double>& bot_position) {
        if (center_points_.empty()) {
            return 0.0; // No center points loaded
        }

        // Find the closest center point to the bot's current position
        double min_distance = std::numeric_limits<double>::max();
        size_t closest_index = 0;
        for (size_t i = 0; i < center_points_.size(); ++i) {
            double distance = calculateDistance2D(bot_position, center_points_[i]);
            if (distance < min_distance) {
                min_distance = distance;
                closest_index = i;
            }
        }

        // Compute the progression as a ratio of the closest index to the total number of points
        return static_cast<double>(closest_index) / static_cast<double>(center_points_.size());
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr progress_publisher_;
    std::string center_points_path_;
    std::vector<std::pair<double, double>> center_points_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrackProgressionNode>());
    rclcpp::shutdown();
    return 0;
}
