from raftstereo import run_model
import cv2

def get_disparity(left_img, right_img):
    """
    Gets the disparity map and saves the npy file to
    Args:
        - left_img(uint8): the left rgb image
        - right_img(uint8): the right rgb image
    """ 
    save_images(left_img, right_img)
    run_model.run_model("image_pngs/left_image.png", "image_pngs/right_image.png") # generates map and outputs to raftstereo/demo_output/images.npy

def save_images(left_img, right_img):
    cv2.imwrite("image_pngs/left.png", left_img)
    cv2.imwrite("images_pngs/right.png", right_img)