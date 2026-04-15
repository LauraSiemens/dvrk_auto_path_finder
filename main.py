#camera parameters
baseline = 20 #in mm
start_pos = [-1.45341, -0.03949, 0.706] #in m

import get_disparity, image_to_coordinates
import cv2
import rclpy
from rclpy.node import Node
import numpy as np
import os


def main():
    rclpy.init()
    node = image_to_coordinates.PathPublisher()
    
    #gets rgb images from coppelia in numpy arrays
    left_img, right_img = image_to_coordinates.get_images()
    
    #dont have this function yet but we need it
    #if disparity map exists in demo_output folder, load it, otherwise generate it and save it to the folder
    if os.path.exists('raftstereo/demo_output/images.npy'):
        disparity_map = np.load('raftstereo/demo_output/images.npy')
    else:
        disparity_map=get_disparity.get_disparity(left_img, right_img) # only generates map
    node.start_pos = start_pos
    node.baseline = baseline
    node.disparity_map = disparity_map
    node.rgb_image_L = left_img    
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()