from raftstereo import run_model

def get_disparity(left_img, right_img):
    """
    Gets the disparity map and saves the npy file to
    Args:
        - left_img(uint8): the left rgb image
        - right_img(uint8): the right rgb image
    """ 
    save_images(left_img, right_img)
    disparity_map = run_model("images/left.png", "images/right.png")

def save_images(left_img, right_img):
    cv2.imsave("images/left.png", left_img)
    cv2.imsave("images/right.png", right_img)