#camera parameters
baseline = 20 #in mm
start_pos = [-1.45341, -0.03949, 0.706] #in m

import get_disparity, image_to_coordinates
import cv2

def main():
    #gets rgb images from coppelia in numpy arrays
    left_img, right_img = image_to_coordinates.get_images()
    
    #dont have this function yet but we need it
    disparity_map=get_disparity.get_disparity(left_img, right_img) # only generates map
    image_to_coordinates(start_pos, disparity_map, left_img)
    



if __name__ == '__main__':
    main()