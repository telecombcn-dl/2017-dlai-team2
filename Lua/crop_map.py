#crop_map.py
def crop_map( image , xi, xf, yi, yf):

	#"Crops the map out of the whole screenshot"
	print(image)
	map_img = image[xi:xf][yi:yf]

	return [map_img]