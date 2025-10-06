#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64.hpp"


// TheJudge node continuously evaluates the robot's performance by processing perception data and episode results, updating a reward system accordingly, 
// and periodically publishing the current evaluation to influence future robot behavior.

class TheJudge : public rclcpp::Node
{
public:
    TheJudge() : Node("the_judge"), total_reward_(0.0), last_update_time_(this->now())
    {
        // Subscription to the bot_perception topic to get the boundary weights, trajectory, and velocity
        perception_subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "bot_perception", 10, std::bind(&TheJudge::perceptionCallback, this, std::placeholders::_1));

        // Subscription to the result topic to get the result status
        result_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "result", 10, std::bind(&TheJudge::resultCallback, this, std::placeholders::_1));

        // Publisher for the total reward/penalty
        judgment_publisher_ = this->create_publisher<std_msgs::msg::Float64>("judgement", 10);

        // Timer to publish judgment at 10 Hz
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&TheJudge::publishJudgment, this));

        // Timer to penalize every 2 seconds
        penalty_timer_ = this->create_wall_timer(
            std::chrono::seconds(2), std::bind(&TheJudge::penalizeOverTime, this));
    }

private:
    void perceptionCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        // Data structure: [left boundary weight, right boundary weight, trajectory, velocity]
        double trajectory = msg->data[2];
        double velocity = msg->data[3];

        double normalized_velocity_value;
        double normalized_trajectory_value;

        // Velocity based reward/penalty
        if (velocity < 35.0)
        {
            normalized_velocity_value = -(velocity / 35.0);  // Penalty as negative value
        }
        else
        {
            normalized_velocity_value = 1.0 - (velocity / 35.0)*10;  // Reward as positive value
        }

        // Trajectory based reward/penalty
        if (trajectory > 30.0)
        {
            normalized_trajectory_value = -((trajectory - 30.0) / 90.0);  // Penalty as negative value
        }
        else
        {
            normalized_trajectory_value = trajectory*5 / 90.0;  // Reward as positive value
        }

        // Update total reward
        total_reward_ = normalized_velocity_value + normalized_trajectory_value;
        // RCLCPP_INFO(this->get_logger(), "Velocity Reward/Penalty: %f", normalized_velocity_value);
        // RCLCPP_INFO(this->get_logger(), "Trajectory Reward/Penalty: %f", normalized_trajectory_value);
    }

    void resultCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        double reward = 0.0;

        if (msg->data == "succeed!")
        {
            reward = 15.0;
        }
        else if (msg->data == "crashed")
        {
            reward = -10.0;
        }
        else if (msg->data == "lost")
        {
            reward = -20.0;
        }

        // Update total reward based on result
        total_reward_ += reward;
        RCLCPP_INFO(this->get_logger(), "Episodic Reward/Penalty: %f", reward);
        
        // Publish the final judgment for the episode
        publishJudgment();

        // Reset the total reward for the next episode
        total_reward_ = 0.0;
    }

    void penalizeOverTime()
    {
        auto current_time = this->now();
        auto elapsed_time = (current_time - last_update_time_).seconds();

        if (elapsed_time >= 2.0)
        {
            total_reward_ -= 1.0;  // Penalize by -1
            last_update_time_ = current_time;  // Reset the time
        }
    }

    void publishJudgment()
    {
        std_msgs::msg::Float64 judgment_msg;
        judgment_msg.data = total_reward_;

        judgment_publisher_->publish(judgment_msg);

        RCLCPP_INFO(this->get_logger(), "Total Judgment Published: %f", total_reward_);
    }

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr perception_subscription_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr result_subscription_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr judgment_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr penalty_timer_;

    double total_reward_;
    rclcpp::Time last_update_time_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TheJudge>());
    rclcpp::shutdown();
    return 0;
}
