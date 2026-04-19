from raftstereo import run_model
import cv2
import numpy as np

def get_disparity(left_img_dir, right_img_dir):
    """
    Gets the disparity map and saves the npy file to
    Args:
        - left_img_dir(string): the left rgb image directory
        - right_img_dir(string): the right rgb image directory
    """ 
    run_model.run_model(left_img_dir, right_img_dir) # generates map and outputs to raftstereo/demo_output/images.npy
    if not "raftstereo/demo_output/disparity.npy":
        print("Disparity map not found. Please check if the RAFT-Stereo model ran successfully.")
        return None
    disparity_map = np.load('raftstereo/demo_output/disparity.npy')
    return disparity_map
