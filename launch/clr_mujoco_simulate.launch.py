import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import xacro


from launch import LaunchDescription
from launch.substitutions import (
    Command,
    FindExecutable,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    clr_mujoco_package_name = "clr_mujoco_config"
    clr_mujoco_package_path = get_package_share_directory(clr_mujoco_package_name)

    model_path = os.path.join(clr_mujoco_package_path,
                                 "description",
                                 "scene.xml")

    clr_mujoco_simulate = Node(
        package='mujoco_ros2_simulation',
        executable='mujoco_simulate',
        output='both',
        parameters=[{"model_path": model_path}],
    )

    return LaunchDescription([
        clr_mujoco_simulate,
    ])
