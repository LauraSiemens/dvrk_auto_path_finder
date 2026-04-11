#camera parameters
baseline = 20 #in mm
start_pos = 

import get_image, get_disparity, image_to_coordinates


def main():
    #gets rgb images from coppelia in numpy arrays
    left_img, right_img = get_image()
    #dont have this function yet but we need it
    disparity_map=get_disparity(left_img, right_img)
    image_to_coordinates(start_pos, disparity_map, left_img)
    



if __name__ == '__main__':
    main()