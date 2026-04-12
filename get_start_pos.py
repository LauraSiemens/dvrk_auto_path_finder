def get_start_pos(world_pos):
    return cam_coord_to_pixel(world_coord_to_cam_coord(world_pos))
def cam_coord_to_pixel(cam_coord):
    """
    takes pixel coordinate and disparity map and gives the corresponding camera coordinate in meters
    Args:
        mac_coord(tuple): some point in work frame (x,y,z)
    Returns: coordinate (x,y) in pixels as an np.array 
    """
    f_x = 924.2773797458503
    f_y = 519.9060261070408

    c_x = 640.0
    c_y = 360.0

    X, Y, Z = cam_coord

    u = (X * f_x / Z) + c_x
    v = c_y - (Y * f_y / Z)

    return (int(u), int(v))

def world_coord_to_cam_coord(world_coord):
    """
    takes world coordinate and gives the corresponding camera coordinate in meters
    Args:
        world_coord(tuple): some point in world frame (x,y,z)
    Returns: coordinate (x,y) in pixels as an np.array 
    """
    qx = -0.662965337815252
    qy = -0.6403711031178091
    qz = -0.26785100317691796
    qw = -0.28045971412006343

    tx = -1.5620065838424029
    ty = -0.004288581503827216
    tz = 0.8126686641282106
    R = tf_transformations.quaternion_matrix([qx, qy, qz, qw])[:3, :3]
    t = np.array([tx, ty, tz])

    return R.T @ (world_coord - t)   
