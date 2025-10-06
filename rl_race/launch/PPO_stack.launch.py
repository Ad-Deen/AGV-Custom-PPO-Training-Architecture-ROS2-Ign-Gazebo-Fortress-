#!/usr/bin/python3

from os.path import join
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    package_name = 'rl_race'
    world_file_name = 'empty.sdf'
    urdf_file_name = 'F1race.urdf'
    
    package_dir = get_package_share_directory(package_name)

    
    # Env setup for world control
    PPO_env_reset = Node(
        package='rl_race',
        executable='PPO_env_reset.py',
        name='pose_reset_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # PPO Policy node
    PPO_policy_ANN = Node(
        package='rl_race',
        executable='PPO_policy_ANN.py',
        name='ppo_policy_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # PPO actor node
    PPO_actor = Node(
        package='rl_race',
        executable='PPO_actor.py',
        name='actor_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # PPO reward function node
    PPO_reward = Node(
        package='rl_race',
        executable='PPO_reward_func.py',
        name='reward_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # PPO update function node
    PPO_update = Node(
        package='rl_race',
        executable='PPO_update.py',
        name='update_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # PPO node to calculate value function for advantage calc
    PPO_value_funcNN = Node(
        package='rl_race',
        executable='PPO_value_funcNN.py',
        name='value_funcNN',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # PPO node to calculate value function for advantage calc
    PPO_policy_funcNN = Node(
        package='rl_race',
        executable='PPO_policy_funcNN.py',
        name='policy_funcNN',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # PPO node to calculate value function for advantage calc
    PPO_policy_loss = Node(
        package='rl_race',
        executable='PPO_loss.py',
        name='PPO_loss_calc',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    

    # PPO node to collect episodic data
    data_collect = Node(
        package='rl_race',
        executable='episodic_data_collector.py',
        name='episodic_data_collector',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # PPO node to visualize policy
    PPO_policy_viz = Node(
        package='rl_race',
        executable='PPO_policy_viz.py',
        name='PPO_policy_viz',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # PPO node to visualize learning
    learning_viz = Node(
        package='rl_race',
        executable='learning_viz.py',
        name='learning_viz',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation (Gazebo) clock if true'),
        PPO_env_reset,                  #resets the env , collisions and manages episode with flags
        # PPO_policy_funcNN,            #CNN policy network of PPO , generalize states , predict action , take advantage, 
                                        #calculate policy loss & update, and predict again.
        PPO_policy_ANN,                 #ANN policy network------------
        # PPO_actor,                      #takes action probs choose based on probs and gives action taken feedback
        # data_collect                  #-------------
        # PPO_value_funcNN,             #predicts value estimates, create advantage, calc value loss , backprop state estimation NN , value estimate again
        # PPO_policy_loss,              #-------------
        PPO_policy_viz,
        learning_viz

    ])
