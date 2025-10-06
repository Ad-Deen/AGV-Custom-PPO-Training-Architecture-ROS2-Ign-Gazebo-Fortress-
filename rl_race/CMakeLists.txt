cmake_minimum_required(VERSION 3.8)
project(rl_race)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)

# Add executables
add_executable(lane_perception src/lane_perception.cpp)
ament_target_dependencies(lane_perception rclcpp std_msgs geometry_msgs sensor_msgs)

add_executable(q_learn_policy src/q_learn_policy.cpp)
ament_target_dependencies(q_learn_policy rclcpp std_msgs geometry_msgs)

add_executable(collision_checker src/collision_checker.cpp)
ament_target_dependencies(collision_checker rclcpp std_msgs sensor_msgs geometry_msgs)

add_executable(the_judge src/the_judge.cpp)
ament_target_dependencies(the_judge rclcpp std_msgs)

add_executable(bot_perception_node src/bot_perception_node.cpp)
ament_target_dependencies(bot_perception_node rclcpp sensor_msgs std_msgs)

add_executable(center_line_plotter src/center_line_plotter.cpp)
ament_target_dependencies(center_line_plotter rclcpp std_msgs nav_msgs sensor_msgs geometry_msgs)

add_executable(track_progression src/track_progression.cpp)
ament_target_dependencies(track_progression rclcpp std_msgs nav_msgs sensor_msgs geometry_msgs)

# Install targets together
install(TARGETS
  q_learn_policy
  lane_perception
  collision_checker
  the_judge
  bot_perception_node
  center_line_plotter
  track_progression
  DESTINATION lib/${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
  scripts/visualize_odometry.py
  scripts/teleop_twist.py
  scripts/maping.py
  scripts/organizer.py
  scripts/set_pose_service.py
  scripts/PPO_env_reset.py
  scripts/PPO_policy_ANN.py
  scripts/PPO_actor.py
  scripts/PPO_reward_func.py
  scripts/PPO_value_funcNN.py
  scripts/img_perception_viz.py
  scripts/PPO_policy_CNN.py
  scripts/PPO_loss.py
  scripts/1D2D.py
  scripts/episodic_data_collector.py
  scripts/center_line_viz.py
  scripts/PPO_policy_viz.py
  scripts/learning_viz.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install other directories (launch files, URDF, etc.)
install(DIRECTORY
  launch urdf worlds scripts
  DESTINATION share/${PROJECT_NAME}/
)

# Testing (optional)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_lint_auto_find_test_dependencies TRUE)
  # Skipping linter checks (remove comments when ready for production)
  # set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
