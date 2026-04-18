from transforms3d.quaternions import quat2mat
import numpy as np
def world_to_pixel(world_pos):
    return cam_coord_to_pixel(world_coord_to_cam_coord(world_pos))
    X_world =  [0.036361380777597985, 0.6988448578557066, 0.7143484546330199, -1.5620065838424029, 0.999330520966789, -0.022534398094745223, -0.0288220534791242, -0.004288581503827216, -0.004044731411661673, 0.7149182229815831, -0.699196377705622, 0.8126686641282106]

    X_cam = R @ X_world + t

    X, Y, Z = X_cam

    if Z <= 0:
        return None

    u = fx * X / Z + cx
    v = cy - fy * Y / Z

    return np.array([int(u), int(v)])
