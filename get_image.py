import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np

class StereoVisionReceiver(Node):
    def __init__(self):
        super().__init__('stereo_vision_receiver')
        self.get_logger().info("Starting Stereo Vision Synchronizer...")

        # The tool that converts ROS2 images to OpenCV matrices
        self.bridge = CvBridge()

        # 1. Create message filter subscribers (These don't trigger callbacks on their own)
        self.left_img_sub = message_filters.Subscriber(self, Image, '/stereo/left/image_raw')
        self.left_info_sub = message_filters.Subscriber(self, CameraInfo, '/stereo/left/camera_info')
        self.right_img_sub = message_filters.Subscriber(self, Image, '/stereo/right/image_raw')
        self.right_info_sub = message_filters.Subscriber(self, CameraInfo, '/stereo/right/camera_info')

        # 2. Synchronize the topics based on their exact timestamps
        # Queue size of 10 is usually plenty for simulation
        self.ts = message_filters.TimeSynchronizer(
            [self.left_img_sub, self.left_info_sub, self.right_img_sub, self.right_info_sub], 
            queue_size=10
        )
        
        # 3. Register the single callback for when a perfect set of 4 arrives
        self.ts.registerCallback(self.stereo_callback)
        self.get_logger().info("Waiting for synchronized stereo data from CoppeliaSim...")

    def stereo_callback(self, left_img_msg, left_info_msg, right_img_msg, right_info_msg):
        # This function only runs when a perfectly synchronized left/right pair is received
        timestamp = left_img_msg.header.stamp.sec + (left_img_msg.header.stamp.nanosec * 1e-9)
        self.get_logger().info(f"Received Synchronized Stereo Pair at t={timestamp:.2f}s")

        try:
            # Convert ROS2 messages to OpenCV format (BGR8 is standard for OpenCV)
            cv_left = self.bridge.imgmsg_to_cv2(left_img_msg, desired_encoding='bgr8')
            cv_right = self.bridge.imgmsg_to_cv2(right_img_msg, desired_encoding='bgr8')
            
            cv_left = cv2.flip(cv_left, 0) 
            cv_right = cv2.flip(cv_right, 0)
            
            #saves image
            
            cv2.
            cv2.waitKey(1) # Required to refresh the OpenCV GUI

        except Exception as e:
            self.get_logger().error(f"Failed to process images: {e}")

def get_image(args=None):
    rclpy.init(args=args)
    node = StereoVisionReceiver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()