import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import math
from PIL import Image
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from tf2_ros import Buffer, TransformListener
import tf_transformations
import get_disparity

class PathPublisher(Node):

    def __init__(self):
        super().__init__('path_publisher')

        # Publisher to the topic your Lua script is listening to
        self.publisher = self.create_publisher(
            Pose,
            'next_coordinate',
            10
        )
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # define start_pos, disparity_map, and rgb_image_L
        rgb_image_L, rgb_image_R = get_images()
        disparity_map = get_disparity(rgb_image_L, rgb_image_R)
        start_pos = (-1.45344,-0.03942,0.706) #defined point to start as defined in scene. position is in world frame coordinates
        
        self.image_to_coordinates(start_pos, disparity_map, rgb_image_L)

    def image_to_coordinates(self, start_pos, disparity_map, rgb_image_L):
        """
        Takes image from coppelia and coverts next point in the path to coordinates
        Args:
            start_pos (tuple): starting postion of end effector in pixels (x,y)
            disparity_map(np.double): The disparity map from depth detection
            rgb_image_L(np.uint8): Image from the coppelia camera
        """
        path_img_L = self.get_threshold_image(rgb_image_L)
        start_coord_L = tuple(sum(coord) for coord in zip(start_pos, (10, 10)))
        all_coords_L = [start_pos]
        all_coords_L.append(self.get_next_coord(path_img_L, start_coord_L, start_pos))
        while all_coords_L[-1] != (0, 0):
            next_coord = self.get_next_coord(path_img_L, all_coords_L[-2], all_coords_L[-1])
            all_coords_L.append(next_coord)
            next_world_coord = self.cam_coord_to_world_coord(self.pixel_to_cam_coord(next_coord, disparity_map))
            pnt = Point(x=next_world_coord[0], y=next_world_coord[1], z=next_world_coord[2])
            qtrn = Quaternion()
            pose = Pose(position=pnt, orientation=qtrn)
            self.publisher.publish(pose)
            self.get_logger().info(f"Publishing: {pose}")

    def get_threshold_image(self, img_rgb):

        """
        makes image black and white
        Args: 
            img_rgb (np.uint8): The image from coppelia
        Returns: black and white image of coppelia camera view (np.uint8)
        """
        diff_allowed = 10
        copied_img = img_rgb.copy().astype(np.uint8)
        

        for i in range(copied_img.shape[0]):
            for j in range(copied_img.shape[1]):
                if (abs(copied_img[i, j, 0] - copied_img[i, j, 1]) > diff_allowed
                    or abs(copied_img[i, j, 0] - copied_img[i, j, 2]) > diff_allowed
                    or abs(copied_img[i, j, 1] - copied_img[i, j, 2]) > diff_allowed):
                    copied_img[i, j, 0] = 0
                    copied_img[i, j, 1] = 0
                    copied_img[i, j, 2] = 0
                else:
                    copied_img[i, j, 0] = 255
                    copied_img[i, j, 1] = 255
                    copied_img[i, j, 2] = 255
        return copied_img.astype(np.uint8)

    def get_next_coord(self, image, last_coord, current_coord):
        """
        Finds the next coord in the image to move to
        Args:
            image(np.uint8): Black and white image of the path
            last_coord(tuple): tuple of the previous coord of end effector (x,y)
            current_coord(tuple): tuple of the current coord of end effector (x,y)
        Returns: tuple of next coordinate
        """
        min_dist = 15
        max_dist = 25

        current_direction = (current_coord[0] - last_coord[0], current_coord[1] - last_coord[1])
        current_angle = np.arctan2(current_direction[1], current_direction[0])

        coord_opt_1 = (0, 0)
        coord_opt_2 = (0, 0)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j, 0] == 255:
                    distance = np.linalg.norm(np.array((j, i)) - np.array(current_coord))
                    new_direction = (j - current_coord[0], i - current_coord[1])
                    new_angle = np.arctan2(new_direction[1], new_direction[0])
                    if distance >= min_dist and distance <= max_dist and (new_angle > current_angle - np.pi/2 and new_angle < current_angle + np.pi/2):
                        if coord_opt_1 == (0, 0):
                            coord_opt_1 = (j, i)
                        else:
                            coord_opt_2 = (j, i)

        next_coord = np.average((np.array(coord_opt_1), np.array(coord_opt_2)), axis=0)
        next_coord = np.round(next_coord).astype(int)
        return tuple(next_coord)

    def pixel_to_cam_coord(pixel_coord, disparity):
        """
        takes pixel coordinate and disparity map and gives the corresponding camera coordinate in meters
        Args:
            pixel_coord(tuple): tuple of some point on the image (in pixels)
            disparity(np.double): The disparity map from depth detection
        Returns: coordinate (x,y,z) in meters in the camera frame as an np.array 
        """
        #focal length in pixels
        f_x = 924.2773797458503
        f_y = 519.9060261070408
        #image centre
        c_x = 640.0
        c_y = 360.0
        baseline = 0.020 # in m
        Z = baseline * f_x / abs(disparity[pixel_coord[1], pixel_coord[0]])
        X = (pixel_coord[0] - c_x) * Z / f_x
        Y = -(pixel_coord[1] - c_y) * Z / f_y
        return (np.array((X,Y,Z)))

    def cam_coord_to_world_coord(cam_coord):
        """
        takes camera coordinate and converts to the corresponding world coordinate
        Args:
            cam_coord(np.array): array representing a point (x,y,z) in the camera frame, in meters
        Returns: coordinate (x,y,z) in meters in the world frame as an np.array 
        """

        qx = -0.662965337815252
        qy = -0.6403711031178091
        qz = -0.26785100317691796
        qw = -0.28045971412006343

        tx = -1.5620065838424029
        ty = -0.004288581503827216
        tz = 0.8126686641282106

        R = tf_transformations.quaternion_matrix([qx, qy, qz, qw])[:3, :3]
        t = np.array([tx, ty, tz])

        return np.dot(R, cam_coord) + t

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
    # Display the images using OpenCV
    cv2.imshow('Left Camera', cv_left)
    cv2.imshow('Right Camera', cv_right)
    stereo_receiver.destroy_node()
    rclpy.shutdown()
    
    return cv_left, cv_right

def main():
    rclpy.init()
    node = PathPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


