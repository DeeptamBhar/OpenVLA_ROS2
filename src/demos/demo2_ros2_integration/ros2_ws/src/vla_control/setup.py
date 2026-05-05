from setuptools import setup
import os
from glob import glob

package_name = 'vla_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Advanced OpenVLA ROS2 control',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vla_controller = vla_control.vla_controller_node:main',
            'task_manager = vla_control.task_manager_node:main',
            'camera_publisher = vla_control.camera_publisher_node:main',
            'performance_monitor = vla_control.performance_monitor_node:main',
        ],
    },
)
