
import numpy as np

import matplotlib.pyplot as plt
from get_start_pos import world_to_pixel
import cv2
start_pos = [-1.45341, -0.03949, 0.706] 
pixel_start_pos = world_to_pixel(start_pos)
print(pixel_start_pos)
img = cv2.imread("image_pngs/left_image.png")
print(img.shape)
radius = 5
color = (0, 0, 255)
thickness = 5
copied_img = img.copy()
cv2.circle(copied_img, pixel_start_pos, radius, color, thickness)
cv2.imshow('Where on path', copied_img)

# 3. Wait for any key press (0 means wait indefinitely)
cv2.waitKey(0)

# 4. Clean up and close the window
cv2.destroyAllWindows()