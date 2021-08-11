# import the necessary packages
import numpy as np
import cv2
#from input_retrieval import *
#python data to sheet
import xlwt
from xlwt import Workbook
import datetime
#plotting the trajectory and bounding box
from matplotlib import image
from matplotlib import pyplot as plt
import time

print(1e-8)
print(np.log(10))

def find_scale(x_image,y_image):
	print("enter width of road and length of road ,num of lanes(visible in image), divider width")
	road_width = input("road width = \n")
	road_length = input("road length = \n")
	#num_lanes = input("num of lane total = \n")
	div_width = input("divider length =\n")
	road_width = float(road_width)
	road_length = float(road_length)
	#num_lanes = int(num_lanes)
	div_width = float(div_width)
	#road_width = road_width - div_width
	vertical_scale = np.square(x_image[1]-x_image[2]) + np.square(y_image[1]-y_image[2])
	vertical_scale = np.sqrt(vertical_scale)
	vertical_scale = np.max(y_image) - np.min(y_image)
	#horizontal_scale = np.square(x_image[1]-x_image[2]) + np.square(y_image[1]-y_image[2])
	H1 = np.square(x_image[0]-x_image[1]) + np.square(y_image[0]-y_image[1])
	H2 = np.square(x_image[2]-x_image[3]) + np.square(y_image[2]-y_image[3])
	#H = grad_H*[]
	SH1 = float(road_width/(np.sqrt(H1) + 1e-8))
	SH2 = float(road_width/(np.sqrt(H2) + 1e-8))
	SV = float(road_length/(vertical_scale + 1e-8))
	S_D = SH1 - SH2

	return SH1, S_D, SV, vertical_scale


videoStream = cv2.VideoCapture('C:\\aa\\vehicle_tracking_college\\tracking\\yolo_youtube\\delhi_dataset.mp4')
print("FPS of video ",videoStream.get(cv2.CAP_PROP_FPS))
ret, frame = videoStream.read()
image_ref = frame

'''
def draw_coordinates(event, x, y, flag, params):
	if event == cv2.EVENT_LBUTTONDBLCLK:
		x_image.append(x)
		y_image.append(y)
		cv2.circle(image_ref, (x,y), 10, (0,255,0), -1)
		if len(x_image) >= 2:
			cv2.line(image_ref, (x_image[-1],y_image[-1]), (x_image[-2], y_image[-2]), (255, 0, 0), 3)
x_image, y_image = [], []
cv2.namedWindow("image_ref")
cv2.setMouseCallback("image_ref",draw_coordinates)
while (1):
    cv2.imshow('image_ref',image_ref)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(x_image,y_image)

cv2.destroyAllWindows()
#calculating scale
scale_hori = []
scale = find_scale(x_image,y_image)
lim = len(y_image)
for i in range(0,lim):
    centerY = y_image[i]
    scale_H = 0
    Hori_scale = scale[1] * (centerY - np.min(y_image)) / (scale[3] + 1e-8)
    scale_H = scale[0] - Hori_scale
    scale_hori.append(scale_H)


print("scale_hori = ",scale_hori)
'''
