import subprocess

def get_disparity(left_img, right_img):
    """
    Gets the disparity map and saves the npy file to
    Args:
        - left_img(uint8): the left rgb image
        - right_img(uint8): the right rgb image
    """ 
    save_images(left_img, right_img)
    subprocess.run([
        "python", "raftsterao/demo.py",
        "--restore_ckpt", "models/raftstereo-middlebury.pth",
        "--corr_implementation", "alt",
        "--save_numpy",
        "--mixed_precision",
        "-l", "../images/left.png",
        "-r", "../images/right.png"
    ])

def save_images(left_img, right_img):
    cv2.imsave("images/left.png", left_img)
    cv2.imsave("images/right.png", right_img)