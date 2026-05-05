"""
Camera Publisher Node - Publishes images from various sources
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
from pathlib import Path


class CameraPublisherNode(Node):
    """
    Publishes camera images from USB, file, or mock source
    """
    
    def __init__(self):
        super().__init__('camera_publisher')
        
        # Parameters
        self.declare_parameter('config_path', '../../../config/camera_config.yaml')
        self.declare_parameter('source_type', 'mock')  # 'usb', 'mock', 'video'
        self.declare_parameter('publish_rate', 10.0)
        
        config_path = self.get_parameter('config_path').value
        source_type = self.get_parameter('source_type').value
        publish_rate = self.get_parameter('publish_rate').value
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Publisher
        topic = self.config['camera']['topics']['input']
        self.publisher = self.create_publisher(Image, topic, 10)
        
        # Initialize source
        self.cap = None
        self.current_frame = None
        
        if source_type == 'usb':
            device = self.config['camera']['usb']['device']
            self.cap = cv2.VideoCapture(device)
            if not self.cap.isOpened():
                self.get_logger().error(f'Failed to open USB camera: {device}')
                return
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['usb']['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['usb']['height'])
            self.get_logger().info(f'Opened USB camera: {device}')
        
        elif source_type == 'mock':
            image_path = self.config['camera']['mock']['image_path']
            self.current_frame = cv2.imread(image_path)
            if self.current_frame is None:
                self.get_logger().error(f'Failed to load image: {image_path}')
                return
            self.get_logger().info(f'Loaded mock image: {image_path}')
        
        elif source_type == 'video':
            self.get_logger().error('Video source not yet implemented')
            return
        
        # Create timer
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_image)
        self.get_logger().info(f'Publishing to {topic} at {publish_rate} Hz')
    
    def publish_image(self):
        """Publish current frame"""
        frame = None
        
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn('Failed to read frame from camera')
                return
        elif self.current_frame is not None:
            frame = self.current_frame.copy()
        
        if frame is None:
            return
        
        # Convert to ROS message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        
        self.publisher.publish(msg)
    
    def destroy_node(self):
        """Cleanup"""
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()