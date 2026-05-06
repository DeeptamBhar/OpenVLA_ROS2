"""
VLA Controller Node - Advanced ROS2 integration with task switching
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import sys
from pathlib import Path
import torch

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent.parent))
from openvla_core.model_manager import ModelManager


class VLAControllerNode(Node):
    """
    Advanced VLA Controller with multi-task support and performance monitoring
    """
    
    def __init__(self):
        super().__init__('vla_controller')
        
        # Parameters
        self.declare_parameter('model_config', '../../../config/model_config.yaml')
        self.declare_parameter('tasks_config', '../../../config/tasks.yaml')
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('max_queue_size', 2)
        self.declare_parameter('enable_visualization', True)
        
        # Get parameters
        model_config_path = self.get_parameter('model_config').value
        tasks_config_path = self.get_parameter('tasks_config').value
        camera_topic = self.get_parameter('camera_topic').value
        max_queue_size = self.get_parameter('max_queue_size').value
        
        self.get_logger().info('Initializing VLA Controller Node...')
        
        # Load tasks configuration
        with open(tasks_config_path, 'r') as f:
            tasks_config = yaml.safe_load(f)
            self.tasks = tasks_config['tasks']
            self.current_task = tasks_config.get('default_task', list(self.tasks.keys())[0])
        
        self.get_logger().info(f'Loaded {len(self.tasks)} tasks')
        self.get_logger().info(f'Current task: {self.current_task}')
        
        # Initialize model manager
        self.get_logger().info('Loading VLA model...')
        self.model_manager = ModelManager(model_config_path)
        if not self.model_manager.load_model():
            self.get_logger().error('Failed to load model!')
            raise RuntimeError('Model loading failed')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            max_queue_size
        )
        
        self.task_sub = self.create_subscription(
            String,
            '/vla/task_command',
            self.task_callback,
            10
        )
        
        # Publishers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/vla/waypoint_markers',
            10
        )
        
        self.annotated_image_pub = self.create_publisher(
            Image,
            '/vla/annotated_image',
            10
        )
        
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            '/vla/predicted_action',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/vla/status',
            10
        )
        
        self.metrics_pub = self.create_publisher(
            String,
            '/vla/metrics',
            10
        )
        
        # State
        self.processing = False
        self.last_action = None
        self.frame_count = 0
        self.dropped_frames = 0
        
        # Performance tracking timer
        self.create_timer(5.0, self.publish_metrics)
        
        self.get_logger().info('VLA Controller Node ready!')
        self.get_logger().info(f'Subscribed to: {camera_topic}')
        self.get_logger().info(f'Publishing to: /vla/waypoint_markers, /vla/annotated_image')
    
    def task_callback(self, msg: String):
        """Handle task switching commands"""
        task_cmd = msg.data.strip()
        
        # Parse command (format: "switch:<task_name>" or "set_params:<key>=<value>")
        if task_cmd.startswith('switch:'):
            new_task = task_cmd.split(':', 1)[1]
            if new_task in self.tasks:
                self.current_task = new_task
                self.get_logger().info(f'Switched to task: {new_task}')
                self.publish_status(f'Task switched to: {new_task}')
            else:
                self.get_logger().warn(f'Unknown task: {new_task}')
                self.publish_status(f'Error: Unknown task {new_task}')
        
        elif task_cmd.startswith('list_tasks'):
            task_list = ', '.join(self.tasks.keys())
            self.get_logger().info(f'Available tasks: {task_list}')
            self.publish_status(f'Available tasks: {task_list}')
        
        elif task_cmd.startswith('current_task'):
            self.publish_status(f'Current task: {self.current_task}')
        
        else:
            self.get_logger().warn(f'Unknown command: {task_cmd}')
    
    def image_callback(self, msg: Image):
        """Process incoming images"""
        self.frame_count += 1
        
        # Drop frames if still processing
        if self.processing:
            self.dropped_frames += 1
            if self.dropped_frames % 10 == 0:
                self.get_logger().warn(
                    f'Dropped {self.dropped_frames} frames due to slow inference'
                )
            return
        
        self.processing = True
        
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Convert to PIL for model
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # Get current task
            task = self.tasks[self.current_task]
            
            # Create prompt (with example substitutions)
            prompt = task['prompt'].format(
                object="red cup",
                container="blue bin",
                target="left",
                drawer="top"
            )
            
            # Run inference
            result = self.model_manager.infer(pil_image, prompt, task=self.current_task)
            
            if result is None:
                self.get_logger().error('Inference failed')
                self.processing = False
                return
            
            # Parse action (this is simplified - real parsing depends on model output format)
            self.last_action = self.parse_action(result['action'])
            
            # Publish action
            self.publish_action(self.last_action)
            
            # Publish visualization
            if self.get_parameter('enable_visualization').value:
                self.publish_markers(self.last_action)
                self.publish_annotated_image(cv_image, self.last_action, result)
            
            # Log performance
            self.get_logger().info(
                f'Processed frame {self.frame_count} in {result["inference_time"]*1000:.1f}ms'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')
        
        finally:
            self.processing = False
    
    def parse_action(self, action_str: str) -> np.ndarray:
        """
        Parse action string to numpy array
        This is a simplified version - adjust based on actual model output
        """
        # For demo purposes, generate random actions
        # In production, parse the actual model output
        task = self.tasks[self.current_task]
        action_dims = task['action_dims']
        
        # Example: Random action in normalized space [-1, 1]
        action = np.random.uniform(-1, 1, action_dims)
        
        return action
    
    def publish_action(self, action: np.ndarray):
        """Publish predicted action"""
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.action_pub.publish(msg)
    
    def publish_markers(self, action: np.ndarray):
        """Publish 3D markers for RViz"""
        marker_array = MarkerArray()
        
        # Create arrow marker for end-effector target
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "vla_waypoints"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Position (scale from normalized [-1,1] to world coordinates)
        marker.pose.position.x = float(action[0] * 0.5)  # Scale to 0.5m range
        marker.pose.position.y = float(action[1] * 0.5)
        marker.pose.position.z = float(action[2] * 0.5 + 0.5)  # Offset z
        
        # Orientation (if available)
        if len(action) >= 6:
            from scipy.spatial.transform import Rotation
            r = Rotation.from_euler('xyz', action[3:6])
            quat = r.as_quat()
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
        else:
            marker.pose.orientation.w = 1.0
        
        # Scale and color
        marker.scale.x = 0.2  # Arrow length
        marker.scale.y = 0.02  # Arrow width
        marker.scale.z = 0.02  # Arrow height
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.lifetime.sec = 0  # Persist until replaced
        
        marker_array.markers.append(marker)
        
        # Add text label
        text_marker = Marker()
        text_marker.header = marker.header
        text_marker.ns = "vla_labels"
        text_marker.id = 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = marker.pose.position.x
        text_marker.pose.position.y = marker.pose.position.y
        text_marker.pose.position.z = marker.pose.position.z + 0.1
        text_marker.scale.z = 0.05
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = self.current_task
        
        marker_array.markers.append(text_marker)
        
        self.marker_pub.publish(marker_array)
    
    def publish_annotated_image(self, cv_image: np.ndarray, action: np.ndarray, result: dict):
        """Publish annotated 2D image"""
        annotated = cv_image.copy()
        h, w = annotated.shape[:2]
        
        # Draw target point (project 3D to 2D - simplified)
        target_x = int((action[0] + 1) * w / 2)
        target_y = int((action[1] + 1) * h / 2)
        
        # Draw crosshair
        cv2.circle(annotated, (target_x, target_y), 10, (0, 255, 0), 2)
        cv2.line(annotated, (target_x - 15, target_y), (target_x + 15, target_y), (0, 255, 0), 2)
        cv2.line(annotated, (target_x, target_y - 15), (target_x, target_y + 15), (0, 255, 0), 2)
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated, f'Task: {self.current_task}', (10, 30), 
                    font, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f'Inference: {result["inference_time"]*1000:.1f}ms', 
                    (10, 60), font, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f'Target: ({action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f})', 
                    (10, 90), font, 0.6, (255, 255, 255), 1)
        
        # Convert back to ROS message
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        annotated_msg.header.stamp = self.get_clock().now().to_msg()
        annotated_msg.header.frame_id = 'camera'
        
        self.annotated_image_pub.publish(annotated_msg)
    
    def publish_status(self, status: str):
        """Publish status message"""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
    
    def publish_metrics(self):
        """Publish performance metrics"""
        metrics = self.model_manager.get_metrics()
        metrics['dropped_frames'] = self.dropped_frames
        metrics['total_frames'] = self.frame_count
        metrics['drop_rate'] = self.dropped_frames / max(self.frame_count, 1)
        
        msg = String()
        msg.data = yaml.dump(metrics)
        self.metrics_pub.publish(msg)
        
        # Log summary
        self.get_logger().info(
            f'Metrics: {metrics["fps"]:.2f} FPS, '
            f'{metrics["drop_rate"]*100:.1f}% dropped, '
            f'{metrics["avg_inference_time"]*1000:.1f}ms avg'
        )


def main(args=None):
    rclpy.init(args=args)
    node = VLAControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down...')
        node.model_manager.print_metrics()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()