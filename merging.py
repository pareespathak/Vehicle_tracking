# import the necessary packages
import numpy as np
import imutils
import time
from scipy import spatial
import cv2
#from input_retrieval import *
#python data to sheet
import xlwt
from xlwt import Workbook
import datetime
#plotting the trajectory and bounding box
from matplotlib import image
from matplotlib import pyplot as plt

# Workbook is created
wb = Workbook()
# add_sheet is used to create sheet.
sheet1 = wb.add_sheet("sheet 1", cell_overwrite_ok=True)
sheet1.write(0, 1, 'id')
sheet1.write(0, 2, 'frame_no')
sheet1.write(0, 3, 'x')
sheet1.write(0, 4, 'y')
#sheet1.write(0, 5, 'type')
sheet1.write(0, 5, 'type and confidence')

# Define constants
# CONF_THRESHOLD is confidence threshold. Only detection with confidence greater than this will be retained
# NMS_THRESHOLD is used for non-max suppression
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4
list_of_vehicles = ["car","bus","truck", "train"]
X_plot,Y_plot = [],[]
# PURPOSE: Draw all the detection boxes with a green dot at the center
# RETURN: N/A
def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame,y_image,num_frames,sr_no):
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			#print("class id:", classIDs[i],"position",x ,y)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			#Draw a green dot in the middle of the box
			cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (0, 0xFF, 0), thickness=2)
			#condition for tracking object and appending in sheet
			#object must be in the bounding area
			#if y >= y_image[0] and y+(h//2) <= y_image[2]:
			if y >= 100 and y <= 800:
				sheet1.write(sr_no,2,num_frames)
				#print(num_frames,i,sr_no)
				sheet1.write(sr_no,1,int(i))
				sheet1.write(sr_no,3,(x + (w//2)))
				sheet1.write(sr_no,4,(y + (h//2)))
				#sheet1.write(sr_no,5,"{}".format(LABELS[classIDs[i]]))
				sheet1.write(sr_no,5,text)
				sr_no = sr_no + 1
	return sr_no

def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
	centerX, centerY, width, height = current_box
	dist = np.inf #Initializing the minimum distance
	# Iterating through all the k-dimensional trees
	for i in range(FRAMES_BEFORE_CURRENT):
		coordinate_list = list(previous_frame_detections[i].keys())
		if len(coordinate_list) == 0: # When there are no detections in the previous frame
			continue
		# Finding the distance to the closest point and the index
		temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
		if (temp_dist < dist):
			dist = temp_dist
			frame_num = i
			coord = coordinate_list[index[0]]

	if (dist > (max(width, height)/2)):
		return False

	# Keeping the vehicle ID constant
	current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
	return True
def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame, X_plot, Y_plot):
	current_detections = {}
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			centerX = x + (w//2)
			centerY = y+ (h//2)

			# When the detection is in the list of vehicles, AND
			# it crosses the line AND
			# the ID of the detection is not present in the vehicles
			#print(vehicle_count, str(datetime.datetime.now()), text)
			if (LABELS[classIDs[i]] in list_of_vehicles):
				current_detections[(centerX, centerY)] = vehicle_count
				if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)):
					vehicle_count += 1
					# vehicle_crossed_line_flag += True
				# else: #ID assigning
					#Add the current detection mid-point of box to the list of detected items
				# Get the ID corresponding to the current detection

				ID = current_detections.get((centerX, centerY))
				# If there are two detections having the same ID due to being too close,
				# then assign a new ID to current detection.
				if (list(current_detections.values()).count(ID) > 1):
					current_detections[(centerX, centerY)] = vehicle_count
					vehicle_count += 1

				#Display the ID at the center of the box
				cv2.putText(frame, str(ID), (centerX, centerY),\
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)
			if i == 4:
				X_plot.append(x + (w//2))
				Y_plot.append(y + (h//2))
				#print(str(ID))
				#sheet1.write()

	return vehicle_count, current_detections

# PURPOSE: Displays the vehicle count on the top-left corner of the frame
# PARAMETERS: Frame on which the count is displayed, the count number of vehicles
# RETURN: N/A
def displayVehicleCount(frame, vehicle_count):
	cv2.putText(
		frame, #Image
		'Detected Vehicles: ' + str(vehicle_count), #Label
		(20, 20), #Position
		cv2.FONT_HERSHEY_SIMPLEX, #Font
		0.8, #Size
		(0, 0xFF, 0), #Color
		2, #Thickness
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)

# Draw a prediction box with confidence and title
def draw_prediction(frame, classes, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))
    cx = (left + right)//2
    cy = (top + bottom)//2
    #print(cx,cy)
    cv2.circle(frame, (cx,cy), 1, (255, 0, 0), 2)
    # Assign confidence to label
    label = '%.2f' % conf
    # Print a label of class.
    if classes:
        assert(classId < len(classes))
        label = '%s: %s' % (classes[classId], label)
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    #print(label)
def process_frame(frame, outs, classes, confThreshold, nmsThreshold,video_width,video_height, vehicle_count, num_frames, X_plot, Y_plot,y_image,sr_no):
	frameHeight = video_height
	frameWidth = video_width
	x1_line = 0
	y1_line = video_height//2
	x2_line = video_width//3
	y2_line = video_height//2
	classIds = []
	confidences = []
	boxes = []
	vehicle_crossed_line_flag = False
	for out in outs:
		for detection in out:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = scores[classId]
			if confidence > confThreshold:
				#scale the detected coordinates to frame original width nd frameHeight
				center_x = int(detection[0] * frameWidth)
				center_y = int(detection[1] * frameHeight)
				width = int(detection[2] * frameWidth)
				height = int(detection[3] * frameHeight)
				left = int(center_x - width/2)
				top = int(center_y - height/2)
				box = np.array([center_x, center_y, width, height])

				classIds.append(classId)
				confidences.append(float(confidence))
				boxes.append([left, top, width, height])
	'''
	if vehicle_crossed_line_flag:
		cv2.line(frame, (x1_line, y1_line),( x2_line, y2_line),(0, 0xFF, 0), 2)
	else:
		cv2.line(frame, (x1_line, y1_line),(x2_line, y2_line), (0, 0, 0xFF), 2)
	'''
	indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
	sr_no = drawDetectionBoxes(indices, boxes, classIds, confidences, frame,y_image,num_frames,sr_no)
	vehicle_count, current_detections = count_vehicles(indices, boxes, classIds, vehicle_count, previous_frame_detections, frame, X_plot, Y_plot)
	return current_detections, previous_frame_detections, vehicle_count, sr_no
	'''
	for i in indices:
	    i = i[0]
	    box = boxes[i]
	    left = box[0]
	    top = box[1]
	    width = box[2]
	    height = box[3]
	    draw_prediction(frame, classes, classIds[i], confidences[i], left, top, left + width, top + height)
	cv2.imshow('img',frame)
	'''
def find_scale(x_image,y_image):
	print("enter width of road and length of road ,num of lanes(visible in image), divider width")
	road_width = input("road width = \n")
	road_length = input("road length = \n")
	num_lanes = input("num of lane total = \n")
	div_width = input("divider length =\n")
	road_width = float(road_width)
	road_length = float(road_length)
	num_lanes = int(num_lanes)
	div_width = float(div_width)
	vertical_scale = np.square(x_image[1]-x_image[2]) + np.square(y_image[1]-y_image[2])
	vertical_scale = np.sqrt(vertical_scale)
	vertical_scale = vertical_scale/(road_length + 1e-8)
	horizontal_scale = np.square(x_image[1]-x_image[2]) + np.square(y_image[1]-y_image[2])
	print(road_width, road_length, num_lanes)
	return 0


# Read COCO dataset classes
with open('cocos.names', 'rt') as f:
	classes = f.read().rstrip('\n').split('\n')
	LABELS = classes
print(classes)

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416


# Load the networO-SeqCNNSLAMk with YOLOv3 weights and config using darknet framework
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg", "darknet")
#for gpu setup
#yet to be done (cuda crash)
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#catpure videoStream
videoStream = cv2.VideoCapture('C:\\Users\\paree\\Downloads\\_imagis\\bridge.mp4')
# initialize the video stream, pointer to output video file, and
# frame dimensions
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
#drawing coordinates for image
ret, frame = videoStream.read()
image_ref = frame
print("Draw coordinates on image and press escape after 4 coordinates")

#x_image, y_image = [0,0,0,0], [0,0,0,0]

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

#calculating scale
#scale = find_scale(x_image,y_image)

#Initialization
previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]
# previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees
num_frames, vehicle_count, sr_no = 0, 0, 2
print("starting tracker")
while( ret == True):
	num_frames = num_frames + 1
	print(num_frames)
	vehicle_crossed_line_flag = False
	ret, frame = videoStream.read()
	x = video_height
	y = video_height
	if not ret:
		break
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight), swapRB=True, crop = False)
	#get output layer
	outNames = net.getUnconnectedOutLayersNames()
	net.setInput(blob)
	outs = net.forward(outNames)
	current_detections, previous_frame_detections, vehicle_count, sr_no = process_frame(frame, outs, classes, CONF_THRESHOLD, NMS_THRESHOLD,
		video_width, video_height, vehicle_count, num_frames,X_plot, Y_plot,y_image,sr_no)
	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	#update previous frame detection
	previous_frame_detections.pop(0)
	#print(type(current_detections))
	#previous_frame_detections.append(spatial.KDTree(current_detections))
	previous_frame_detections.append(current_detections)

#print(X_plot, Y_plot)
#X_plot = np.array(X_plot)
#Y_plot = np.array(Y_plot)
plt.scatter(X_plot, Y_plot, s = 10)
plt.imshow(image_ref)
plt.show()
#print(X_plot)
#print(Y_plot)
#cv2.waitKey(0)
cv2.destroyAllWindows()
wb.save('data1.xls')
print("end")
