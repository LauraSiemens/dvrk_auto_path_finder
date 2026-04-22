import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import math
import yaml
from PIL import Image
from pathlib import Path
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from tf2_ros import Buffer, TransformListener
from get_disparity import get_disparity
from transforms3d.quaternions import quat2mat
from get_start_pos import world_to_pixel
from scipy.spatial.transform import Rotation as R

with open('scene_parameters.yaml', 'r') as file:
    config = yaml.safe_load(file)

camera_matrix = config['camera_matrix']
baseline = config['baseline']
#focal legnths
fx = config['fx']
fy = config['fy']
#camera centre
cx = config['cx']
cy = config['cy']



class PathPublisher(Node):

    def __init__(self):
        super().__init__('path_publisher')

        # Publisher to the topic your Lua script is listening to
        self.publisher = self.create_publisher(
            PoseArray,
            'path_coordinates',
            10
        )
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        start_pos = [-1.45341, -0.03949, 0.706] #in m
        print("Start pos in world coordinates: ", start_pos)
        start_pos = world_to_pixel(start_pos)
        print("\nStart pos in pixel coordinates: ", start_pos)
        left_img, right_img, left_path, right_path = get_images() 
        # print directory contents of image_pngs to check if the images were saved correctly
        
        #if disparity map exists in demo_output folder, load it, otherwise generate it and save it to the folder
        if os.path.exists('demo_output/disparity.npy'):
           disparity_map = np.load('demo_output/disparity.npy')
        else:

            ## TO RUN THIS NEXT LINE YOU NEED TO GO TO raftstereo/run_model.py and change the restore_ckpt and output_directory paths to your local paths where you saved the model and where you want the output to be saved. 
            # I have it set to my local paths but you need to change jobbinport to your username in those paths.
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            disparity_map = get_disparity(left_img, right_img) # only generates map from the two png files(saved in get_images) and saves it to raftstereo/demo_output/images.npy
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    

        start_pos = start_pos
        disparity_map = disparity_map
        rgb_image_L = left_img    
        self.image_to_coordinates(start_pos, disparity_map, rgb_image_L)

    def image_to_coordinates(self, start_pos, disparity_map, rgb_image_L):
        """
        Takes image from coppelia and coverts next point in the path to coordinates
        Args:
            start_pos (tuple): starting postion of end effector in pixels (x,y)
            disparity_map(np.double): The disparity map from depth detection
            rgb_image_L(np.uint8): Image from the coppelia camera
        """
        print('Getting path...\n')
        path_img_L = self.get_threshold_image(rgb_image_L)
        start_coord_L = tuple(sum(coord) for coord in zip(start_pos, (10, 10)))
        all_coords_L = [start_pos]
        all_coords_L.append(self.get_next_coord(path_img_L, start_coord_L, start_pos))
        next_world_coord = self.cam_coord_to_world_coord(self.pixel_to_cam_coord(start_pos, disparity_map))
        last_world_coord = self.cam_coord_to_world_coord(self.pixel_to_cam_coord(start_coord_L, disparity_map))

        # Calculate desired k_vector of end-effector to keep the ring plane perpendicular to wire path
        k_vector = np.array(next_world_coord) - np.array(last_world_coord)
        k_vector = k_vector / np.linalg.norm(k_vector)

        # Calculate quaternion to transform end-effector k_kector to desired k_vector
        rot_matrix, RMSD = R.align_vectors([k_vector], [[0,0,1]])
        quat = rot_matrix.as_quat()

        # Initialize array with pose of first coordinate, in ROS2 Pose message type
        pnt = Point(x=next_world_coord[0], y=next_world_coord[1], z=next_world_coord[2])
        qtrn = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        pose_array = [Pose(position=pnt, orientation=qtrn)]

        while all_coords_L[-1] != (0, 0):
            next_coord = self.get_next_coord(path_img_L, all_coords_L[-2], all_coords_L[-1])
            all_coords_L.append(next_coord)
            last_world_coord = next_world_coord
            next_world_coord = self.cam_coord_to_world_coord(self.pixel_to_cam_coord(next_coord, disparity_map))

            # Calculate desired k_vector of end-effector to keep the ring plane perpendicular to wire path
            k_vector = np.array(next_world_coord) - np.array(last_world_coord)
            k_vector = k_vector / np.linalg.norm(k_vector)

            # Calculate quaternion to transform end-effector k_kector to desired k_vector
            rot_matrix, rmsd = R.align_vectors([k_vector], [[0,0,1]])
            quat = rot_matrix.as_quat()

            # Append current point to array of all poses
            pnt = Point(x=next_world_coord[0], y=next_world_coord[1], z=next_world_coord[2])
            qtrn = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            pose = Pose(position=pnt, orientation=qtrn)
            pose_array.append(pose)
        del all_coords_L[-1] # delete end coordinate that goes to (0,0)
        # Convert array of poses to ROS2 message of type PoseArray to send to CoppeliaSim
        msg = PoseArray()
        msg.poses = pose_array

        self.publisher.publish(msg)
        #self.get_logger().info(f"Publishing: {msg.poses}")

    def get_threshold_image(self, img_rgb):

        """
        makes image black and white
        Args: 
            img_rgb (np.uint8): The image from coppelia
        Returns: black and white image of coppelia camera view (np.uint8)
        """
        diff_allowed = 10
        copied_img = img_rgb.copy().astype(int)
        

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
        Finds the next coord in the image to move to.

        Args:
            image (np.uint8): Black and white image of the path
            last_coord (tuple): previous coordinate (x, y)
            current_coord (tuple): current coordinate (x, y)

        Returns:
            tuple: next coordinate
        """
        min_dist = 10
        max_dist = 20

        cx, cy = current_coord
        lx, ly = last_coord

        # Current direction
        dx0 = cx - lx
        dy0 = cy - ly
        current_angle = np.arctan2(dy0, dx0)

        # Get all white pixels at once
        white_pixels = np.argwhere(image[:, :, 0] == 255)   # rows, cols = y, x

        if white_pixels.size == 0:
            return (0, 0)

        ys = white_pixels[:, 0]
        xs = white_pixels[:, 1]

        # Vector from current point to each candidate
        dx = xs - cx
        dy = ys - cy

        # Distance and agnle masks
        distances = np.sqrt(dx * dx + dy * dy)
        dist_mask = (distances >= min_dist) & (distances <= max_dist)
        angles = np.arctan2(dy, dx)
        angle_diff = np.arctan2(np.sin(angles - current_angle), np.cos(angles - current_angle))
        angle_mask = np.abs(angle_diff) < (np.pi / 3)

        # Combined valid candidates
        valid_mask = dist_mask & angle_mask
        valid_xs = xs[valid_mask]
        valid_ys = ys[valid_mask]

        if len(valid_xs) == 0:
            angles = np.arctan2(dy, dx)
            angle_diff = np.arctan2(np.sin(angles - current_angle), np.cos(angles - current_angle))
            angle_mask = np.abs(angle_diff) < (np.pi / 2)

            # Combined valid candidates
            valid_mask = dist_mask & angle_mask
            valid_xs = xs[valid_mask]
            valid_ys = ys[valid_mask]

            if len(valid_xs) == 0:   
                return (0, 0)

        # Match your old behavior: take first two valid points if available
        if len(valid_xs) == 1:
            return (int(valid_xs[0]), int(valid_ys[0]))

        coord1 = np.array([valid_xs[0], valid_ys[0]])
        coord2 = np.array([valid_xs[-1], valid_ys[-1]])

        next_coord = np.round((coord1 + coord2) / 2).astype(int)
        return tuple(next_coord)

    def pixel_to_cam_coord(self, pixel_coord, disparity):
        """
        takes pixel coordinate and disparity map and gives the corresponding camera coordinate in meters
        Args:
            pixel_coord(tuple): tuple of some point on the image (in pixels)
            disparity(np.double): The disparity map from depth detection
        Returns: coordinate (x,y,z) in meters in the camera frame as an np.array 
        """
        print("Disparity", disparity)

        #focal length in pixels
        f_x = 924.2773797458503
        f_y = 519.9060261070408
        #image centre
        c_x = 640.0
        c_y = 360.0
        baseline = 0.020 # in m
        #create depth with adjusted scalign based on initial condition
        Z = baseline * f_x / (abs(disparity[pixel_coord[1], pixel_coord[0]]))
        X = (pixel_coord[0] - c_x) * Z / f_x
        X *= -1 
        Y = -(pixel_coord[1] - c_y) * Z / f_y
        
        return (np.array((X,Y,Z)))

    def cam_coord_to_world_coord(self, cam_coord):
        """
        takes camera coordinate and converts to the corresponding world coordinate
        Args:
            cam_coord(np.array): array representing a point (x,y,z) in the camera frame, in meters
        Returns: coordinate (x,y,z) in meters in the world frame as an np.array 
        """
        M = np.array(camera_matrix).reshape(3, 4)
    
        R = M[:, :3]
        t = M[:, 3]
        world_pos = R @ cam_coord + t

        return world_pos

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
    """
    Gets images from Coppelia
    """
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

    cv_left_bgr = cv2.cvtColor(cv_left, cv2.COLOR_RGB2BGR)
    cv_right_bgr = cv2.cvtColor(cv_right, cv2.COLOR_RGB2BGR)
    #cv2.imshow('Left Camera', cv_left_bgr)
    #cv2.imshow('Right Camera', cv_right_bgr)
    left_path = "image_pngs/left_image.png"
    right_path = "image_pngs/right_image.png"
    cv2.imwrite(left_path, cv_left_bgr)
    cv2.imwrite(right_path, cv_right_bgr)
    
    return cv_left_bgr, cv_right_bgr, left_path, right_path

def main():
    rclpy.init()
    node = PathPublisher()
    
    try:
        rclpy.spin(node)
        print('Spinning...')
    except KeyboardInterrupt:
        pass
    print('Shutting down...')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


