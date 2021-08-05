# import the necessary packages
import numpy as np
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
import time

# Workbook is created
wb = Workbook()
# add_sheet is used to create sheet.
sheet1 = wb.add_sheet("sheet 1", cell_overwrite_ok=True)
sheet1.write(0, 1, 'id')
sheet1.write(0, 2, 'frame_no')
sheet1.write(0, 3, 'x')
sheet1.write(0, 4, 'y')
sheet1.write(0, 5, 'type and confidence')
sheet1.write(0, 6, 'x_real')
sheet1.write(0, 7, 'y_real')
sheet1.write(0, 8, 'w_fac')
sheet1.write(0, 9, 'y_scale fac')
sheet1.write(0, 10, 'Fps')
sheet1.write(0, 11, 'time')



# Define constants
# CONF_THRESHOLD is confidence threshold. Only detection with confidence greater than this will be retained
# NMS_THRESHOLD is used for non-max suppression
CONF_THRESHOLD = 0.3
#NMS to overcome the multiple boxes over single object
NMS_THRESHOLD = 0.4
list_of_vehicles = ["car","bus","truck", "train"]


# PURPOSE: Draw all the detection boxes with a green dot at the center
# RETURN: N/A
def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame,y_image,x_image,num_frames,sr_no,scale):
	# ensure at least one detection exists
	if len(idxs) > 0:
		#print(idxs)
		# loop over the indices we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# draw a bounding box rectangle and label on the frame
			#Draw a green dot in the middle of the box
			c_x = x + (w//2)
			c_y = y + h
			cv2.circle(frame, (c_x, c_y), 2, (0, 0xFF, 0), thickness=2)

			#condition for tracking object and appending in sheet
			#object must be in the bounding area

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
def count_vehicles(idxs, boxes, classIDs, confidences, vehicle_count, previous_frame_detections, frame, y_image, x_image, num_frames, sr_no, scale, X_plot, Y_plot):
    current_detections = {}
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            centerX = x + (w//2)
            centerY = y + (h//2)

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
                centerY = y + h
                if centerY >= np.min(y_image) and centerY <= np.max(y_image) and centerX >= x_image[0] and centerX <= x_image[1] and LABELS[classIDs[i]] =='car':
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    #print("class id:", classIDs[i],"position",x ,y)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                        confidences[i])
                    cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    text1 = "{}: {:.4f}".format(x,y)
                    cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    #print(num_frames,i,sr_no)
                    #exporting data in sheet
                    sheet1.write(sr_no,1,ID)
                    sheet1.write(sr_no,2,num_frames)
                    sheet1.write(sr_no,3,int(centerX))
                    sheet1.write(sr_no,4,int(centerY))
                    sheet1.write(sr_no,5,"{}".format(LABELS[classIDs[i]]))
                    image_co = np.array([[centerX], [centerY], [1]])
                    #print(type(image_co))
                    real_co = np.dot(scale, image_co)
                    X_real = real_co[0]
                    Y_real = real_co[1]
                    h_real = real_co[2]
                    sheet1.write(sr_no,6,float(X_real/(h_real + 1e-8)))
                    sheet1.write(sr_no,7,float(Y_real/(h_real + 1e-8)))
                    sheet1.write(sr_no,8,float(h_real))
                    if ID == 0:
                        X_plot.append(centerX)
                        Y_plot.append(centerY)
                    #######increment number
                    sr_no = sr_no + 1
                # If there are two detections having the same ID due to being too close,
                # then assign a new ID to current detection.
                if (list(current_detections.values()).count(ID) > 1):
                    current_detections[(centerX, centerY)] = vehicle_count
                    vehicle_count += 1
                #Display the ID at the center of the box
                cv2.putText(frame, str(ID), (centerX, centerY),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)

    return vehicle_count, current_detections, sr_no

# PURPOSE: Displays the vehicle count on the top-left corner of the frame
# PARAMETERS: Frame on which the count is displayed, the count number of vehicles
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


def process_frame(frame, outs, classes, confThreshold, nmsThreshold,video_width,video_height, vehicle_count, num_frames, y_image,x_image,sr_no,scale, X_plot, Y_plot):
	frameHeight = video_height
	frameWidth = video_width
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
	#### optional parameters to create line ( counting blink line)
	if vehicle_crossed_line_flag:
		cv2.line(frame, (x_image[0], y_image[0]), (x_image[1], y_image[1]), (0, 0xFF, 0), 2)
	else:
		cv2.line(frame, (x_image[0], y_image[0]), (x_image[1], y_image[1]), (0, 0, 0xFF), 2)
	'''
	######## indices stores values  Thus exportng indices in vehicle_count function
	######## indices = idxs
	indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
	##### vehicle_count provides the sheet data
	vehicle_count, current_detections, sr_no = count_vehicles(indices, boxes, classIds, confidences, vehicle_count, previous_frame_detections, frame, y_image, x_image, num_frames, sr_no, scale, X_plot, Y_plot)
	##### Draws the center dot for each object
	drawDetectionBoxes(indices, boxes, classIds, confidences, frame,y_image,x_image,num_frames,sr_no,scale)
	return current_detections, previous_frame_detections, vehicle_count, sr_no
'''
def find_scale(x_image,y_image):
	print("enter width of road and length of road in (m)")
	road_width = input("road width = \n")
	road_length = input("road length = \n")
	Y_divs = input("Y_dividion length for scaling =\n")
	Y_divs_L = input("Y_dividion length for scaling lower =\n")
	road_width = float(road_width)
	road_length = float(road_length)
	Y_divs = float(Y_divs)
	Y_divs_L = float(Y_divs_L)
	#vertial distance of ROI
	vertical_scale = np.square(x_image[1]-x_image[2]) + np.square(y_image[1]-y_image[2])
	vertical_scale = np.sqrt(vertical_scale)
	vertical_scale = np.max(y_image) - np.min(y_image)
	#horizontal_scale = np.square(x_image[1]-x_image[2]) + np.square(y_image[1]-y_image[2])
	#V1 = np.square(x_image[0]-x_image[5]) + np.square(y_image[0]-y_image[5])
	#V2 = np.square(x_image[3]-x_image[4]) + np.square(y_image[3]-y_image[4])
	V2 = np.square(y_image[3]-y_image[4]) + np.square(x_image[3]-x_image[4])
	V1 = np.square(y_image[0]-y_image[5]) + np.square(x_image[0]-x_image[5])
	#####################################################################################
	H1 = np.square(x_image[0]-x_image[1]) + np.square(y_image[0]-y_image[1])
	H2 = np.square(x_image[2]-x_image[3]) + np.square(y_image[2]-y_image[3])
	#H = grad_H*[]
	SH1 = float(road_width/(np.sqrt(H1) + 1e-8))         ######3 scale for upper line(ususally larger)
	SH2 = float(road_width/(np.sqrt(H2) + 1e-8))		######### scale for lower line ( ususally amaller )
	SVm = float(road_length/(vertical_scale + 1e-8))
	S_D = SH1 - SH2										####### gradiednt paramter, change the order of SH!, SH2 when required
	#grad_M = S_D * (np.max(y_image)-np.min(y_image)) * 0.5 / (vertical_scale + 1e-8)		###### horizontal scale for middle line
	#SH_M = SH1-grad_M													 ############ same as above
	############################################### approach 2 for y scale
	SV1 = float(Y_divs/(np.sqrt(V1) + 1e-8))
	SV2 = float(Y_divs_L/(np.sqrt(V2) + 1e-8))
	S_DV = SV1-SV2
	#S_DV = 0.025
	#SV1 = 0.0855
	print(SV1,SV2,SVm,S_DV,SH2)


	return SH1, S_D, SVm, vertical_scale, S_DV, SV1, #Y_factor
'''
# Read COCO dataset classes
with open('cocos.names', 'rt') as f:
	classes = f.read().rstrip('\n').split('\n')
	LABELS = classes
#print(classes)

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416


# Load the networO-SeqCNNSLAMk with YOLOv3 weights and config using darknet framework
#net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg", "darknet")
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg", "darknet")
#net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3_tin.cfg", "darknet")
#for gpu setup
cuda = True
if cuda:
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#catpure videoStream
#videoStream = cv2.VideoCapture('C:\\Users\\paree\\Downloads\\_imagis\\bridge.mp4')

dst, src = [], []

videoStream = cv2.VideoCapture('C:\\aa\\vehicle_tracking_college\\tracking\\yolo_youtube\\datasets\\delhi_dataset_Trim.mp4')
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS of video ",videoStream.get(cv2.CAP_PROP_FPS))
#drawing coordinates for image
ret, frame = videoStream.read()
#image_ref = frame
reference_img = frame
def draw_coordinates(event, x, y, flag, params):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        x_image.append(x)
        y_image.append(y)
        cv2.circle(image_ref, (x,y), 10, (255,0,0), -1)
        src.append([x,y])
        if len(x_image) >= 2:
            cv2.line(image_ref, (x_image[-1],y_image[-1]), (x_image[-2], y_image[-2]), (255, 0, 0), 3)
x_image, y_image = [], []
image_ref = frame
cv2.namedWindow("image_ref")
cv2.setMouseCallback("image_ref",draw_coordinates)
while (1):
    cv2.imshow('image_ref',image_ref)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#diagonal_aspect_pixel = np.square(y_image[0]-y_image[2]) + np.square(x_image[0]-x_image[2])
#diagonal_aspect_pixel = int(np.sqrt(diagonal_aspect_pixel))
print(x_image, y_image)
cv2.destroyAllWindows()
dst = [[0.0,0.0], [10.5, 0.0], [10.5, 30.0], [0.0, 30.0], [0.0, 20], [0.0, 10.0]]
#points = np.float32(points[:, np.newaxis, :])
#dst = np.float32(dst[:, np.newaxis, :])
#src = np.float32(src[:, np.newaxis, :])
############### replacing x_image and y_image in main frames  ########################
print(src)
print(dst)
homography_mat, Mask = cv2.findHomography(np.float32(src), np.float32(dst), method = cv2.RANSAC)
print(type(homography_mat), "\n", homography_mat.shape)
image_co = np.array([[556.0], [186.0], [1.0]])
#print(type(image_co))
real_co = np.dot(homography_mat, image_co)
#print(real_co, real_co[1])
#calculating scale
#scale = find_scale(x_image,y_image)
#print(scale)
scale = homography_mat
X_plot = []
Y_plot = []
#Initialization
previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]
# previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees
num_frames, vehicle_count, sr_no = 0, 0, 2
print("starting tracker")

while( ret == True):
	start_time = time.time() 						##start time of loop
	num_frames = num_frames + 1
	#print(num_frames)
	vehicle_crossed_line_flag = False
	ret, frame = videoStream.read()
	if not ret:
		break
	#print(frame)
	#ret = False
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight), swapRB=True, crop = False)
	#get output layer
	outNames = net.getUnconnectedOutLayersNames()
	net.setInput(blob)
	outs = net.forward(outNames)
	current_detections, previous_frame_detections, vehicle_count, sr_no = process_frame(frame, outs, classes, CONF_THRESHOLD, NMS_THRESHOLD,
		video_width, video_height, vehicle_count, num_frames, y_image, x_image, sr_no, scale, X_plot, Y_plot)
	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	#update previous frame detection
	previous_frame_detections.pop(0)
	#print(type(current_detections))
	#previous_frame_detections.append(spatial.KDTree(current_detections))
	previous_frame_detections.append(current_detections)

	Fps = float(time.time() - start_time)           #Processing frame per seconds calculation
	#print("Fps = ", 1/Fps)
	#sheet1.write(sr_no,10,float(1/Fps))
	#sheet1.write(sr_no,11,float(Fps))


plt.imshow(image_ref)
plt.plot(X_plot,Y_plot)
plt.show()
cv2.destroyAllWindows()
######## output file name
wb.save('co_transform_5_8a.xls')
print("end")
