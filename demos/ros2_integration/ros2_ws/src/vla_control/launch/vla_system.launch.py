"""
Launch file for complete VLA system
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    
    # Declare arguments
    source_type_arg = DeclareLaunchArgument(
        'source_type',
        default_value='mock',
        description='Camera source type: usb, mock, or video'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='5.0',
        description='Camera publish rate in Hz'
    )
    
    enable_viz_arg = DeclareLaunchArgument(
        'enable_visualization',
        default_value='true',
        description='Enable visualization outputs'
    )
    
    # Nodes
    camera_node = Node(
        package='vla_control',
        executable='camera_publisher',
        name='camera_publisher',
        parameters=[{
            'source_type': LaunchConfiguration('source_type'),
            'publish_rate': LaunchConfiguration('publish_rate'),
        }],
        output='screen'
    )
    
    vla_node = Node(
        package='vla_control',
        executable='vla_controller',
        name='vla_controller',
        parameters=[{
            'enable_visualization': LaunchConfiguration('enable_visualization'),
        }],
        output='screen'
    )
    
    return LaunchDescription([
        source_type_arg,
        publish_rate_arg,
        enable_viz_arg,
        camera_node,
        vla_node,
    ])