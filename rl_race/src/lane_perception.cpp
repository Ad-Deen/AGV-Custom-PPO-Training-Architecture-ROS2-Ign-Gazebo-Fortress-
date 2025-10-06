#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/string.hpp>
#include <vector>
#include <cmath>    // For sin and cos functions
#include <sstream>  // For std::ostringstream

// The provided code implements a ROS 2 node named LanePerceptionNode that processes LiDAR data to convert it 
// from polar (spherical) to Cartesian coordinates.


class LanePerceptionNode : public rclcpp::Node
{
public:
    LanePerceptionNode()
        : Node("lane_perception_node")
    {
        // Subscribe to /scan topic
        subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&LanePerceptionNode::scan_callback, this, std::placeholders::_1));
        
        // Publisher for the formatted string message
        perception_publisher_ = this->create_publisher<std_msgs::msg::String>("/img_perception", 10);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr perception_publisher_;

    // Callback function to process laser scan data
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        int total_samples = msg->ranges.size();

        if (total_samples < 48)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough LiDAR samples.");
            return;
        }

        std::ostringstream oss;
        oss << "["; // Start the string with an opening bracket

        // Convert LiDAR data from spherical to Cartesian coordinates
        for (int i = 0; i < total_samples; ++i)
        {
            float angle = msg->angle_min + i * msg->angle_increment;
            float range = msg->ranges[i];

            // Check if the range is valid
            if (std::isfinite(range) && range < msg->range_max && range > msg->range_min)
            {
                // Convert to Cartesian coordinates
                float x = range * cos(angle);
                float y = range * sin(angle);

                // Append the coordinates as a tuple to the string stream
                oss << "(" << x << "," << y << ")";
                
                // Add a comma and space after each tuple except the last one
                if (i < total_samples - 1) {
                    oss << ", ";
                }
            }
        }

        oss << "]"; // End the string with a closing bracket

        // Convert the string stream to a standard string
        std::string perception_string = oss.str();

        // Create a String message and assign the formatted string to it
        std_msgs::msg::String perception_msg;
        perception_msg.data = perception_string;

        // Publish the string message
        perception_publisher_->publish(perception_msg);
    }
};

int main(int argc, char *argv[])
{
    // Initialize the ROS2 system
    rclcpp::init(argc, argv);

    // Create and spin the node
    rclcpp::spin(std::make_shared<LanePerceptionNode>());

    // Shutdown the ROS2 system
    rclcpp::shutdown();
    return 0;
}