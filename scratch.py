import numpy as np
import matplotlib.pyplot as plt
import pickle


dat = np.load('/home/pmateosaparicio/data/Repository/ESPM154/2025-07-04_04_ESPM154/reconstruction/video_timeline.npy')
print(1/np.mean(np.diff(dat)))
with open('/home/pmateosaparicio/data/Repository/ESPM154/2025-07-04_06_ESPM154/recordings/dlcEyeRight.pickle', "rb") as file:
    eyedat = pickle.load(file)
x=0
mask_edge = np.load('/home/pmateosaparicio/data/Repository/ESPM154/2025-07-04_04_ESPM154/reconstruction/mask_edges.npy')
print(1/np.mean(np.diff(dat)))

import numpy as np
import cv2

# Load your shape points
points = np.load("/home/pmateosaparicio/data/Repository/ESPM154/2025-07-04_04_ESPM154/reconstruction/mask_shape.npy")  # shape: (N, 2)

# Normalize or scale points if necessary
points = points.astype(np.int32)

# Determine the canvas size
x_max, y_max = points[:, 0].max(), points[:, 1].max()
x_min, y_min = points[:, 0].min(), points[:, 1].min()

width, height = x_max - x_min + 1, y_max - y_min + 1

# Shift points so they start at (0,0)
points_shifted = np.column_stack((points[:, 0] - x_min, points[:, 1] - y_min))

# Create a blank mask
mask = np.zeros((height, width), dtype=np.uint8)

# Fill the shape
cv2.fillPoly(mask, [points_shifted], 255)

# Find the outline (contour)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Extract the contour coordinates
outline = contours[0].squeeze()  # shape: (M, 2)

# Shift back to original coordinate system
outline += np.array([x_min, y_min])

# Save the outline
np.save("/home/pmateosaparicio/data/Repository/ESPM154/2025-07-04_04_ESPM154/reconstruction/mask_edges.npy", outline)
