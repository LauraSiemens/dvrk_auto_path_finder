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

        # The tool that converts ROS2 images to OpenCV matrices
        self.bridge = CvBridge()
        self.left_img = None
        self.right_img = None

        # 1. Create message filter subscribers (These don't trigger callbacks on their own)
        self.left_img_sub = message_filters.Subscriber(self, Image, '/stereo/left/image_raw')
        self.right_img_sub = message_filters.Subscriber(self, Image, '/stereo/right/image_raw')

        # 2. Synchronize the topics based on their exact timestamps
        # Queue size of 10 is usually plenty for simulation
        self.ts = message_filters.TimeSynchronizer(
            [self.left_img_sub, self.right_img_sub], 
            queue_size=10
        )
        
        # 3. Register the single callback for when a perfect set of 4 arrives
        self.ts.registerCallback(self.stereo_callback)

    def stereo_callback(self, left_img_msg, right_img_msg):
        # This function only runs when a perfectly synchronized left/right pair is received        
        self.left_img = left_img_msg
        self.right_img = right_img_msg
        
def get_images(args=None):
    if not rclpy.ok():
        rclpy.init(args=args)
    stereo_receiver = StereoVisionReceiver()
    cv_left, cv_right, left_info, right_info = None, None, None, None
    while stereo_receiver.left_img is None or stereo_receiver.right_img is None:
        rclpy.spin_once(stereo_receiver)  

    # Convert ROS2 messages to OpenCV format (BGR8 is standard for OpenCV)
    cv_left = stereo_receiver.bridge.imgmsg_to_cv2(stereo_receiver.left_img, desired_encoding='rgb8')
    cv_right = stereo_receiver.bridge.imgmsg_to_cv2(stereo_receiver.right_img, desired_encoding='rgb8')
    
    cv_left = cv2.flip(cv_left, 0) 
    cv_right = cv2.flip(cv_right, 0)
    ## Display the images using OpenCV
    #cv2.imshow('Left Camera', cv_left)
    #cv2.imshow('Right Camera', cv_right)
    stereo_receiver.destroy_node()
    rclpy.shutdown()
    
    return cv_left, cv_right
    
    return cv_left, cv_right

