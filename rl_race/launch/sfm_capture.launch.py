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
    world_file_name = 'empty_world.sdf'
    urdf_file_name = 'F1race.urdf'
    
    package_dir = get_package_share_directory(package_name)

    
    # Initiate mapping sequence
    visualizier = Node(
        package='rl_race',
        executable='visualize_odometry.py',
        name='visualize_odometry',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation (Gazebo) clock if true'),
        # visualizier,
        # center_line_viz,
    ])
