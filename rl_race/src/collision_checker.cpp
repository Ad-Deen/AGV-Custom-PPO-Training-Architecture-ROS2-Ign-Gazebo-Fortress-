#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/bool.hpp"


//This node checks collisions of any sort with the bot .

class CollisionChecker : public rclcpp::Node
{
public:
    CollisionChecker()
    : Node("collision_checker")
    {
        // Subscriber to LiDAR data
        lidar_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&CollisionChecker::lidar_callback, this, std::placeholders::_1));

        // Publisher for collision detection
        collision_publisher_ = this->create_publisher<std_msgs::msg::Bool>("collisions", 10);

        // Timer for 30Hz refresh rate
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33), std::bind(&CollisionChecker::timer_callback, this));
    }

private:
    void lidar_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        // Calculate the minimum range from the LiDAR data
        min_range_ = *std::min_element(msg->ranges.begin(), msg->ranges.end());
    }

    void timer_callback()
    {
        auto collision_msg = std_msgs::msg::Bool();
        // Check if the minimum range is less than 2.5 meters
        collision_msg.data = (min_range_ < 3.0);

        // Publish the collision status
        collision_publisher_->publish(collision_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_subscription_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr collision_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    float min_range_{std::numeric_limits<float>::infinity()};
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CollisionChecker>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
