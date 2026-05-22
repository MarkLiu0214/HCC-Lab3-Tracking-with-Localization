from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='tracking_with_localization', executable='apriltag_detector_node', name='apriltag_detector_node'),
        Node(package='tracking_with_localization', executable='balloon_detector_node', name='balloon_detector_node'),
        Node(package='tracking_with_localization', executable='tracking_node', name='tracking_node'), 
        Node(package='rviz2', executable='rviz2', name='rviz2'), 
    ])
