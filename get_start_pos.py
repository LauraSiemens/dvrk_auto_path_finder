from transforms3d.quaternions import quat2mat
import numpy as np
def world_to_pixel(world_pos):
    M =  [0.036361380777597985, 0.6988448578557066, 0.7143484546330199, -1.5620065838424029, 0.999330520966789, -0.022534398094745223, -0.0288220534791242, -0.004288581503827216, -0.004044731411661673, 0.7149182229815831, -0.699196377705622, 0.8126686641282106]
    #focal length in pixels
    fx = 924.2773797458503
    fy = 519.9060261070408
    #image centre
    cx = 640.0
    cy = 360.0
    M = np.array(M).reshape(3, 4)
    
    R = M[:, :3]
    t = M[:, 3]

    # Invert transform
    R_wc = R
    t_wc = t

    R_cw = R_wc.T
    t_cw = -R_wc.T @ t_wc

    cam_pos = R_cw @ world_pos + t_cw
    cam_pos[0] *= -1 

    X, Y, Z = cam_pos
    print(X)
    print(Y)
    print(Z)

    if Z <= 0:
        return None

    u = fx * X / Z + cx
    v = cy - fy * Y / Z

    return np.array([int(u), int(v)])