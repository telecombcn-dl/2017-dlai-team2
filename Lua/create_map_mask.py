#create_map_mask.py
import cv2
import numpy as np

def create_map_mask( frame1, frame2, size_SE1, size_SE2 ):

	#"Creates a mask to filter the original grey images in order to display only the circuit with a certain padding around"
	frame1[frame1< 0.8] = 0
	frame2[frame2< 0.8] = 0
	pre_mask = frame1*frame2

	kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size_SE1,size_SE2))


	img_erosion = cv2.erode(img, kernel, iterations=1)
	img_dilation = cv2.dilate(img, kernel, iterations=1)

	cv2.imshow('Dilation', img_dilation)

	#cv2.waitKey(0)

	SE1 = strel('disk',25)
	SE2 = strel('disk',15)
	pre_mask_dil = imdilate(pre_mask,size_SE1)
	mask = imerode(pre_mask,size_SE2)

	cv2.imshow('Mask', mask)

	return [mask]