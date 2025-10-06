#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/float64.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <unordered_map>
#include <vector>
#include <iomanip>  // For std::setprecision


// Input and Output of the QLearningNode
// Input:

// State Space (/bot_perception topic): This node subscribes to the /bot_perception topic to receive std_msgs::msg::Float64MultiArray messages. 
// The message contains a vector of floating-point numbers representing the current state of the environment. 
// The node rounds these state values to a precision of 0.01.

// Rewards (/judgement topic): It subscribes to the /judgement topic to receive std_msgs::msg::Float64 messages, which represent the reward value for the 
// current state-action pair. This reward is used to update the Q-value for that state-action pair.

// Actions (/cmd_vel topic): The node also subscribes to the /cmd_vel topic to receive geometry_msgs::msg::Twist messages. These messages indicate the current 
// action being taken based on the bot's motion commands, such as turning left or right, or adjusting speed (thrust or brake). The actions are mapped to discrete integers (0 = left, 1 = right, 2 = thrust, 3 = brake).

// Output:

// Q-Table (/q_table topic): The node publishes the Q-table data to the /q_table topic as std_msgs::msg::Float64MultiArray messages. The Q-table stores the 
// Q-values (quality values) for each state-action pair, which are updated using the Q-learning algorithm.

// Log Information: Periodically, the node prints the contents of the Q-table to the ROS 2 log, which includes the state, action, and the corresponding 
// Q-value for each state-action pair.


class QLearningNode : public rclcpp::Node
{
public:
    QLearningNode()
        : Node("q_learn"), learning_rate_(0.1)
    {
        // Subscription to the bot_perception topic (state space)
        bot_perception_subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/bot_perception", 10, std::bind(&QLearningNode::botPerceptionCallback, this, std::placeholders::_1));

        // Subscription to the judgment topic (rewards)
        judgment_subscription_ = this->create_subscription<std_msgs::msg::Float64>(
            "/judgement", 10, std::bind(&QLearningNode::judgmentCallback, this, std::placeholders::_1));

        // Subscription to the cmd_vel topic (actions)
        cmd_vel_subscription_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&QLearningNode::cmdVelCallback, this, std::placeholders::_1));

        // Publisher to publish the Q-table
        q_table_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/q_table", 10);

        // Timer to print the Q-table every second (adjust as needed)
        print_timer_ = this->create_wall_timer(
            std::chrono::seconds(1), std::bind(&QLearningNode::printQTable, this));
    }

private:
    using State = std::vector<double>;  // State is a vector of 4 values
    using Action = int;  // 0 = left, 1 = right, 2 = thrust, 3 = brake

    struct StateAction
    {
        State state;
        Action action;

        bool operator==(const StateAction &other) const
        {
            return state == other.state && action == other.action;
        }
    };

    struct StateActionHash
    {
        std::size_t operator()(const StateAction &sa) const
        {
            std::size_t seed = 0;
            for (auto &val : sa.state)
            {
                seed ^= std::hash<double>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            seed ^= std::hash<int>()(sa.action) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };

    std::unordered_map<StateAction, double, StateActionHash> q_table_;
    State current_state_;
    Action current_action_;
    double current_reward_;
    double learning_rate_;

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr bot_perception_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr judgment_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscription_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr q_table_publisher_;
    rclcpp::TimerBase::SharedPtr print_timer_;

    void botPerceptionCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        // Get and round state values to 0.01 precision
        current_state_ = {
            roundToPrecision(msg->data[0], 0.01),
            roundToPrecision(msg->data[1], 0.01),
            roundToPrecision(msg->data[2], 0.01),
            roundToPrecision(msg->data[3], 0.01)
        };
    }

    void judgmentCallback(const std_msgs::msg::Float64::SharedPtr msg)
    {
        current_reward_ = msg->data;

        // Update the Q-value for the current (state, action) pair
        StateAction state_action = {current_state_, current_action_};
        q_table_[state_action] += learning_rate_ * (current_reward_ - q_table_[state_action]);
    }

    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        if (msg->angular.z > 0.0)
        {
            current_action_ = 1;  // Right
        }
        else if (msg->angular.z < 0.0)
        {
            current_action_ = 0;  // Left
        }
        else if (msg->linear.x > 0.0)
        {
            current_action_ = 2;  // Thrust
        }
        else if (msg->linear.x < 0.0)
        {
            current_action_ = 3;  // Brake
        }
    }

    double roundToPrecision(double value, double precision)
    {
        return std::round(value / precision) * precision;
    }

    void printQTable()
    {
        RCLCPP_INFO(this->get_logger(), "Q-Table:");
        for (const auto &entry : q_table_)
        {
            const StateAction &state_action = entry.first;
            double q_value = entry.second;

            std::string state_str = "State: [";
            for (size_t i = 0; i < state_action.state.size(); ++i)
            {
                state_str += std::to_string(state_action.state[i]);
                if (i < state_action.state.size() - 1)
                    state_str += ", ";
            }
            state_str += "]";

            RCLCPP_INFO(this->get_logger(), "%s, Action: %d, Q-Value: %.4f", state_str.c_str(), state_action.action, q_value);
        }
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<QLearningNode>());
    rclcpp::shutdown();
    return 0;
}
