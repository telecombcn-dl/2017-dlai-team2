#pre_processing.py
import cv2
import numpy as np
from crop_map import  crop_map

def pre_processing( image_file, xi, xf, yi, yf ):

	#"Pre-processes the image file"

	#"reads file"
	img = cv2.imread(image_file)

	#"crop"
	map_img = crop_map(img, xi, xf, yi, yf)

	#"rgb 2 gray"
	print(np.array(map_img,dtype=np.uint8))
	map_img_g = cv2.cvtColor( np.int_(map_img), cv2.COLOR_RGB2GRAY )

	#"normalization"
	map_img_g = map_img_g/255

	return [map_img_g]