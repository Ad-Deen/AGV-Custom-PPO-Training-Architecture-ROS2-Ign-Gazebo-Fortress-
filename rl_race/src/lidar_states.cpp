#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <vector>
#include <algorithm>
#include <cmath>  // For sin and cos

// The code provided is for a ROS 2 node named BotPerceptionNode that processes sensor data to compute and publish 
// information about a bot's perception, including track trajectory and velocity.

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
        size_t num_samples = msg->ranges.size();
        if (num_samples < 5)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough LiDAR samples.");
            return;
        }

        std::vector<std::pair<double, double>> range_angle_pairs;

        for (size_t i = 0; i < num_samples; ++i)
        {
            double range = msg->ranges[i];
            double angle = msg->angle_min + i * msg->angle_increment;

            if (std::isfinite(range) && range < msg->range_max && range > msg->range_min)
            {
                range_angle_pairs.emplace_back(range, angle);
            }
        }

        // Sort by range value in descending order and take the top 5
        std::partial_sort(range_angle_pairs.begin(), range_angle_pairs.begin() + 5, range_angle_pairs.end(), std::greater<>());

        // Calculate the mean angle of the top 5 max ranges
        double angle_sum = 0.0;
        for (size_t i = 0; i < 5 && i < range_angle_pairs.size(); ++i)
        {
            angle_sum += range_angle_pairs[i].second;
        }
        track_trajectory_ = angle_sum / 5.0;

        // Publish the updated perception data
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

    double track_trajectory_ = 0.0;  // To store track trajectory
    double velocity_ = 0.0;          // To store velocity
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BotPerceptionNode>());
    rclcpp::shutdown();
    return 0;
}