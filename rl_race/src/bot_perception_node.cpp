#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>  // For sin, cos, and M_PI

//publishes key features for the bot to percieve it's state like velocity and track trajectory angle. 
//Or the angle of the max range where space is open.

class BotPerceptionNode : public rclcpp::Node
{
public:
    BotPerceptionNode() : Node("bot_perception_node")
    {
        // Subscriber to the LIDAR topic
        lidar_subscriber_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&BotPerceptionNode::lidarCallback, this, std::placeholders::_1));

        // Subscriber to the joint states topic
        joint_state_subscriber_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10, std::bind(&BotPerceptionNode::jointStateCallback, this, std::placeholders::_1));

        // Publisher to the bot_perception topic (for track trajectory and velocity)
        perception_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/bot_perception", 10);
    }

private:
    void lidarCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        const double max_lidar_range = 75.0;  // Max range of LIDAR in meters
        const int num_samples = 72;           // Total number of LIDAR samples

        if (msg->ranges.size() != num_samples)
        {
            RCLCPP_WARN(this->get_logger(), "Unexpected number of LIDAR samples: %zu", msg->ranges.size());
            return;
        }

        std::vector<std::pair<double, double>> range_angle_pairs;
        std::vector<double> inf_angles;

        for (size_t i = 0; i < msg->ranges.size(); ++i)
        {
            double range = msg->ranges[i];
            double angle = msg->angle_min + i * msg->angle_increment;

            // Skip invalid range readings
            if (range > max_lidar_range)
            {
                range = max_lidar_range;
            }

            if (std::isinf(range))
            {
                inf_angles.push_back(angle);
                range = max_lidar_range;
            }

            // Store range and angle pairs for processing top 5 max ranges
            range_angle_pairs.emplace_back(range, angle);
        }

        // Sort by range value in descending order and take the top 5
        std::partial_sort(range_angle_pairs.begin(), range_angle_pairs.begin() + 5, range_angle_pairs.end(), std::greater<>());

        // Include inf angles in top 5 max range calculation
        for (double inf_angle : inf_angles)
        {
            range_angle_pairs.push_back({max_lidar_range, inf_angle});
        }

        // Sort again to get top 5
        std::partial_sort(range_angle_pairs.begin(), range_angle_pairs.begin() + 5, range_angle_pairs.end(), std::greater<>());

        // Calculate the mean angle of the top 5 max ranges
        double angle_sum = 0.0;
        for (size_t i = 0; i < 5; ++i)
        {
            angle_sum += range_angle_pairs[i].second;
        }
        track_trajectory_ = angle_sum / 5.0;

        // Publish track trajectory and velocity
        publishPerceptionData();
    }

    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // Find the indices for rear left and rear right wheel joints
        auto rear_left_it = std::find(msg->name.begin(), msg->name.end(), "rear_left_wheel_joint");
        auto rear_right_it = std::find(msg->name.begin(), msg->name.end(), "rear_right_wheel_joint");

        if (rear_left_it == msg->name.end() || rear_right_it == msg->name.end())
        {
            RCLCPP_WARN(this->get_logger(), "Rear wheel joints not found in /joint_states");
            return;
        }

        size_t rear_left_index = std::distance(msg->name.begin(), rear_left_it);
        size_t rear_right_index = std::distance(msg->name.begin(), rear_right_it);

        // Calculate the average velocity of the rear wheels
        double rear_left_velocity = msg->velocity[rear_left_index];
        double rear_right_velocity = msg->velocity[rear_right_index];
        velocity_ = (rear_left_velocity + rear_right_velocity) / 2.0;

        // Publish the updated perception data
        publishPerceptionData();
    }

    void publishPerceptionData()
    {
        // Publish track trajectory and velocity
        std_msgs::msg::Float64MultiArray perception_msg;
        perception_msg.data = {track_trajectory_, velocity_};
        perception_publisher_->publish(perception_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr perception_publisher_;

    double track_trajectory_;  // To store track trajectory
    double velocity_;          // To store velocity
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BotPerceptionNode>());
    rclcpp::shutdown();
    return 0;
}
