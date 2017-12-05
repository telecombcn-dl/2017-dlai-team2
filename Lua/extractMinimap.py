#extractMinimap.py
import cv2
import numpy as np
import pre_processing
from create_map_mask import create_map_mask
from crop_map import  crop_map
from pre_processing import pre_processing

#"image vector"
num_frames = 8 #"number of frames"

frame_vec = []

frame_vec.append(["frame"+str(n+1)+".png" for n in range(num_frames)]) #"create a vector with the frame names"

print(frame_vec[0][0])

#"coordinates of the crop"
xi = 800
xf = 1400
yi = 1800
yf = 2180

#"frame distance between the two images used for the mask creation"
dist = 3

#"prepare the mask"

proc_img_1 = pre_processing (frame_vec[0][1], xi, xf, yi, yf)
proc_img_2 = pre_processing (frame_vec[0][(1+dist)], xi, xf, yi, yf)

mask = create_map_mask(proc_img_1, proc_img_2, 25, 12)

#"mask the frames to isolate the map"

for i in range(num_frames):
	frame_vec[i] = frame_vec[0][i]*mask

cv2.imshow('Masked image example', frame_vec[0][num_frames])
