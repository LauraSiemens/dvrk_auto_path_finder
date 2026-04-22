from transforms3d.quaternions import quat2mat
import numpy as np
import yaml
with open('scene_parameters.yaml', 'r') as file:
    config = yaml.safe_load(file)

camera_matrix = config['camera_matrix']
#focal legnths
fx = config['fx']
fy = config['fy']
#camera centre
cx = config['cx']
cy = config['cy']

def world_to_pixel(world_pos):
    """
    Puts world coordintes into pixel values
    Args:
        world_pos (tuple): X,Y,Z values in coordinates provided by Coppelia
    Returns: tuple of pixel value (x,y)
    """
    M = np.array(camera_matrix).reshape(3, 4)
    
    R = M[:, :3]
    t = M[:, 3]

    # Invert transform
    R_wc = R
    t_wc = t

    R_cw = R_wc.T
    t_cw = -R_wc.T @ t_wc

    cam_pos = R_cw @ world_pos + t_cw
    #flip y axis
    cam_pos[0] *= -1 

    X, Y, Z = cam_pos

    if Z <= 0:
        return None

    u = fx * X / Z + cx
    v = cy - fy * Y / Z

    return np.array([int(u), int(v)])