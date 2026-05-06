# Demo 2: Advanced ROS 2 Integration

Full ROS 2 workspace with multi-task support, performance monitoring, and visualization.

## Features

 **Multi-task switching** - Change tasks at runtime via ROS topics  
 **Performance monitoring** - Real-time metrics publishing  
 **3D visualization** - MarkerArray for RViz  
 **2D annotation** - Annotated camera feed  
 **Modular architecture** - Separate nodes for camera, control, monitoring  
 **Launch files** - Easy system startup

## Quick Start

```bash
cd ros2_ws

# Build
colcon build --symlink-install

# Source
source install/setup.bash

# Launch complete system
ros2 launch vla_control vla_system.launch.py source_type:=mock

# Or run nodes individually
# Terminal 1: Camera
ros2 run vla_control camera_publisher --ros-args -p source_type:=mock

# Terminal 2: VLA Controller
ros2 run vla_control vla_controller

# Terminal 3: RViz
rviz2
```

## Task Switching

```bash
# List available tasks
ros2 topic pub /vla/task_command std_msgs/String "data: 'list_tasks'" --once

# Switch task
ros2 topic pub /vla/task_command std_msgs/String "data: 'switch:push_to_target'" --once

# Check current task
ros2 topic pub /vla/task_command std_msgs/String "data: 'current_task'" --once
```

## RViz Configuration

1. Set Fixed Frame: `map`
2. Add MarkerArray: `/vla/waypoint_markers`
3. Add Image: `/vla/annotated_image`
4. Add Image: `/camera/image_raw`

## Published Topics

- `/vla/waypoint_markers` - 3D action visualization
- `/vla/annotated_image` - 2D annotated camera feed
- `/vla/predicted_action` - Raw action values
- `/vla/status` - Status messages
- `/vla/metrics` - Performance metrics (every 5s)

## Subscribed Topics

- `/camera/image_raw` - Input camera feed
- `/vla/task_command` - Task control commands