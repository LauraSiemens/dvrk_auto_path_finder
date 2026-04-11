import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def image_to_coodinates(start_pos, disparity_map, rgb_image_L):
    """
    Takes image from coppelia and coverts next point in the path to coordinates
    Args:
        start_pos (tuple): starting postion of end effector in pixels (x,y)
        disparity_map(np.double): The disparity map from dpeth detection
        rgb_image_L(np.uint8): Image from the coppelia camera
    """
    path_img_L = get_threshold_image(rgb_image_L)
    start_coord_L = tuple(sum(coord) for coord in zip(start_pos, (10, 10)))
    all_coords_L = [start_pos]
    all_coords_L.append(get_next_coord(path_img_L, start_coord_L, start_pos))
    while all_coords_L[-1] != (0, 0):
    next_coord = get_next_coord(path_img_L, all_coords_L[-2], all_coords_L[-1])
    all_coords_L.append(next_coord)
    next_world_coord = cam_coord_to_world_coord(pixel_to_cam_coord(next_coord, disparity_map))
    # publish next_world_coord as a Pose message

def get_threshold_image(img_rgb):

    """
    makes image black and white
    Args: 
        img_rgb (np.int): The image from coppelia
    Returns: black adn white image of coppelia camera view
    """
    diff_allowed = 10

    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            if (abs(img_rgb[i, j, 0] - img_rgb[i, j, 1]) > diff_allowed
                or abs(img_rgb[i, j, 0] - img_rgb[i, j, 2]) > diff_allowed
                or abs(img_rgb[i, j, 1] - img_rgb[i, j, 2]) > diff_allowed):
                # or img_rgb[i, j, 0] < 60
                # or img_rgb[i, j, 2] > 150):
                img_rgb[i, j, 0] = 0
                img_rgb[i, j, 1] = 0
                img_rgb[i, j, 2] = 0
            else:
                img_rgb[i, j, 0] = 255
                img_rgb[i, j, 1] = 255
                img_rgb[i, j, 2] = 255
    return img_rgb.astype(np.uint8)

def get_next_coord(image, last_coord, current_coord):
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
                distance_last = np.linalg.norm(np.array((j, i)) - np.array(last_coord))
                new_direction = (j - current_coord[0], i - current_coord[1])
                new_angle = np.arctan2(new_direction[1], new_direction[0])
                if distance >= min_dist and distance <= max_dist and (new_angle > current_angle - np.pi/2 and new_angle < current_angle + np.pi/2):
                    if coord_opt_1 == (0, 0):
                        coord_opt_1 = (j, i)
                    else:
                        coord_opt_2 = (j, i)

    next_coord = np.average((np.array(coord_opt_1), np.array(coord_opt_2)), axis=0)
    next_coord = np.round(next_coord).astype(int)
    next_coord = tuple(next_coord)
    return next_coord


def pixel_to_cam_coord(pixel_coord, disparity):
    """
    
    """
    #focal length in pixels
    f_x = 924.2773797458503
    f_y = 519.9060261070408
    #image centre
    c_x = 640.0
    c_y = 360.0
    baseline = 20 # in mm
    Z = baseline * f_x / abs(disparity[pixel_coord])
    X = (pixel_coord[0] - c_x) * Z / f_x
    Y = (pixel_coord[1] - c_y) * Z / f_y
    return(np.array((X,Y,Z)))

def cam_coord_to_world_coord(cam_coord):
    tranformation_matrix = simros2.getObjectMatrix(cameraHandle, sim.handle_inverse)
    R = tranformation_matrix[:3, :3]
    t = tranformation_matrix[:3, 3]
    return np.dot(R, cam_coord) + t
