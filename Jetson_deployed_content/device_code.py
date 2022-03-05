from ast import While
import os
import socket
#from tabnanny import check
import cv2
import base64
import glob
import sys
import time
#import threading
from datetime import datetime
import pickle
from xlwt import Workbook
import csv
import numpy as np
import pytz
# import the necessary packages
from scipy import spatial
#plotting the trajectory and bounding box
#from matplotlib import image
import xlrd
#python data to sheet
import xlwt
import shutil

'''
camera paraeters
'''
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

'''constant paramerters for OD
# CONF_THRESHOLD is confidence threshold. Only detection with confidence greater than this will be retained
# NMS_THRESHOLD is used for non-max suppression
# #NMS to overcome the multiple boxes over single object
'''
UTC = pytz.utc
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4
X_plot, Y_plot = 0, []
'''reading from coco'''
with open('cocos.names', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
    LABELS = classes

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416
# Load the networO-SeqCNNSLAMk with YOLOv3 weights and config using darknet framework
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg", "darknet")
#for gpu setup
cuda = True
if cuda:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

'''Functions used '''
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

# define our clear function
def clear():
  
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')
    
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
                if centerY >= np.min(y_image) and centerY <= np.max(y_image) and centerX >= np.min(x_image) and centerX <= np.max(x_image):
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                        confidences[i])
                    cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    #text1 = "{}: {:.4f}".format(x,y)
                    #exporting data in sheet
                    sheet1.write(sr_no,0,video_num)
                    sheet1.write(sr_no,1,ID)
                    sheet1.write(sr_no,2,num_frames)
                    sheet1.write(sr_no,3,int(centerX))
                    sheet1.write(sr_no,4,int(centerY))
                    sheet1.write(sr_no,5,text)
                    image_co = np.array([[centerX], [centerY], [1]])
                    #print(type(image_co))
                    real_co = np.dot(scale, image_co)
                    X_real = real_co[0]
                    Y_real = real_co[1]
                    h_real = real_co[2]
                    sheet1.write(sr_no,6,float(X_real/(h_real + 1e-8)))
                    sheet1.write(sr_no,7,float(Y_real/(h_real + 1e-8)))
                    #sheet1.write(sr_no,8,float(h_real))
                    #######increment number
                    X_plot = ID
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

def process_frame(frame, outs, classes, confThreshold, nmsThreshold,video_width,video_height, vehicle_count, num_frames, y_image,x_image,sr_no,scale, X_plot, Y_plot):
    frameHeight = video_height
    frameWidth = video_width
    classIds = []
    confidences = []
    boxes = []
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

    ######## indices stores values  Thus exportng indices in vehicle_count function
    ######## indices = idxs
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    ##### vehicle_count provides the sheet data
    vehicle_count, current_detections, sr_no = count_vehicles(indices, boxes, classIds, confidences, vehicle_count, previous_frame_detections, frame, y_image, x_image, num_frames, sr_no, scale, X_plot, Y_plot)
    ##### Draws the center dot for each object
    drawDetectionBoxes(indices, boxes, classIds, confidences, frame,y_image,x_image,num_frames,sr_no,scale)
    return current_detections, previous_frame_detections, vehicle_count, sr_no

def sheet_generator(loc):
    print("into sheets")
    try:
        Fps = 30
        wb = xlrd.open_workbook(loc)
        sheet = wb.sheet_by_index(0)
        ## range = id no present 
        for i in range(0, int(sheet.nrows)):
            # Workbook is created
            wb1 = Workbook()
            # add_sheet is used to create sheet.
            sheet1 = wb1.add_sheet("sheet 1", cell_overwrite_ok=True)
            sheet1.write(0, 1, 'id')
            sheet1.write(0, 2, 'frame_no')
            sheet1.write(0, 3, 'x')
            sheet1.write(0, 4, 'y')
            sheet1.write(0, 5, 'type and confidence')
            sheet1.write(0, 6, 'x_real')
            sheet1.write(0, 7, 'y_real')
            sheet1.write(0, 8, 'Fps')
            sheet1.write(0, 9, 'time')
            sheet1.write(0, 10, 'time_cu')
            sheet1.write(0, 11, 'delta_Y')
            sheet1.write(0, 12, 'insta velo')
            sheet1.write(0, 13, 'Avg velo')


            id_no = i
            name_id = -1
            yp = 0
            yn = 0
            Fp = 0
            Fn = 0
            j = 2
            time_cu = 1e-8
            for k in range(2, int(sheet.nrows)):
                if int(sheet.cell_value(k, 1)) == id_no:
                    name_id = id_no
                    #print(name_id)
                    if j == 2:
                        sheet1.write(j, 1, id_no)
                        sheet1.write(j, 2, sheet.cell_value(k, 2))
                        sheet1.write(j, 3, sheet.cell_value(k, 3))
                        sheet1.write(j, 4, sheet.cell_value(k, 4))
                        sheet1.write(j, 5, sheet.cell_value(k, 5))
                        sheet1.write(j, 6, sheet.cell_value(k, 6))
                        sheet1.write(j, 7, sheet.cell_value(k, 7))
                        Y_B = sheet.cell_value(k, 7)
                        yp = sheet.cell_value(k, 7)
                        Fp = sheet.cell_value(k, 2)
                        sheet1.write(j, 8, Fps)
                        sheet1.write(j, 9, 0)
                        sheet1.write(j, 10, time_cu)
                        sheet1.write(j, 11, 0)
                        sheet1.write(j, 12, 0)
                        sheet1.write(j, 13, 0)
                        j = j + 1
                        print("done")
                    else:
                        sheet1.write(j, 1, id_no)
                        sheet1.write(j, 2, sheet.cell_value(k, 2))
                        sheet1.write(j, 3, sheet.cell_value(k, 3))
                        sheet1.write(j, 4, sheet.cell_value(k, 4))
                        sheet1.write(j, 5, sheet.cell_value(k, 5))
                        sheet1.write(j, 6, sheet.cell_value(k, 6))
                        sheet1.write(j, 7, sheet.cell_value(k, 7))
                        sheet1.write(j, 8, Fps)
                        ### time F2-F1 /Fps
                        yn = sheet.cell_value(k, 7)
                        Fn = sheet.cell_value(k, 2)
                        time = Fn - Fp
                        #print(time)
                        time = float(time/Fps)
                        #print("time",time)
                        ### time cummulative
                        time_cu = time_cu + time
                        #print(time_cu)
                        ############### delta Y
                        delta_Y = yn - yp
                        ############## insta velo
                        if time != 0:
                            insta_velo = float((delta_Y*18)/(time*5))
                        else:
                            insta_velo = 0
                        ########### avg velo
                        avg_velo = yn - Y_B
                        #print(avg_velo)
                        avg_velo = float(avg_velo*18/(time_cu*5))
                        #print("y1", delta_Y)
                        #print("time_cu", time_cu)
                        #print("insta_velo", insta_velo)
                        #print(avg_velo)
                        sheet1.write(j, 9, time)
                        sheet1.write(j, 10, time_cu)
                        sheet1.write(j, 11, delta_Y)
                        sheet1.write(j, 12, insta_velo)
                        sheet1.write(j, 13, avg_velo)
                        Fp = Fn
                        yp = yn
                        j = j + 1

            if name_id != -1:
                save_name = 'sheets/generated_output' + str(name_id)
                wb1.save(save_name + '.xls')
        return False
    except:
        return True

#capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
capture = cv2.VideoCapture('/home/parees/Downloads/Vehicle_tracking-main/Jetson/device/delhi_dataset_Trim.mp4')
ret, frame = capture.read()
cv2.imwrite('detection.jpg', frame)
path_file = 'parameter.txt'
Check = os.path.isfile(path_file)
video_num = 0
daily_check = False
daily_check =True
while True:
    time.sleep(2)
    clear()
    current_date = datetime.now()
    times_now = int(current_date.strftime("%H%M%S"))
    #if  times_now <= 170000 and times_now >= 80000:                    ##### change later 
    if  times_now >= 80000 and times_now < 191200:    ##### change later 
        print("Time satisfied")
        ## creating workbook
        wb = Workbook()
        # add_sheet is used to create sheet.
        sheet1 = wb.add_sheet("sheet 1", cell_overwrite_ok=True)
        sheet1.write(0, 0, 'save name')
        sheet1.write(0, 1, 'id')
        sheet1.write(0, 2, 'frame_no')
        sheet1.write(0, 3, 'x')
        sheet1.write(0, 4, 'y')
        sheet1.write(0, 5, 'type and confidence')
        sheet1.write(0, 6, 'x_real')
        sheet1.write(0, 7, 'y_real')
        #sheet1.write(0, 9, 'y_scale fac')
        sheet1.write(0, 8, 'Fps')
        sheet1.write(0, 11, 'time for velo calculation')
        video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("printing video widths", video_height, video_width)
        '''capture_width=1920,
        capture_height=1080,
        display_width=960,
        display_height=540
        '''
        Video_Fps = capture.get(cv2.CAP_PROP_FPS)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]
        # previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees
        
        num_frames, vehicle_count, sr_no = 0, 0, 2
        print(Check)
        if Check == True:
            try:
                with open(path_file) as f:
                    src = eval(f.readline())
                    dst = eval(f.readline())
                    x_image = eval(f.readline())
                    y_image = eval(f.readline())
                    list_of_vehicles = eval(f.readline())
                    if list_of_vehicles == None:
                        list_of_vehicles = ["car","bus","truck"]
                    email =f.readline()
                    #msg['To'] = email                                    ## Default classes
                if len(src) == len(dst):
                    homography_mat, Mask = cv2.findHomography(np.float32(src), np.float32(dst), method = cv2.RANSAC)
                    scale = homography_mat

                #Frame = 18,000                       
                index_frame = 0        
                #save_name = datetime.now(UTC)
                save_name = video_num
                ret, frame = capture.read()
                #video_num = video_num + 1
                out = cv2.VideoWriter('save_videos/'+str(video_num)+'.mp4', fourcc, 30, (960, 540))
                time_capture = str(datetime.now(UTC))
                sheet1.write(1, 0, save_name)
                '''
                #100000 - 180000
                times_now = int(current_date.strftime("%H%M%S"))
                #if  times_now <= 160000 and times_now >= 80000:
                '''
                times_now = int(current_date.strftime("%H%M%S"))
                ####### conditions from time to break ?
                print("into mail loop")
                #while(ret == True and times_now <= 160000 and times_now >= 80000):     ### change later 
                while(ret == True and times_now >= 80000 and times_now <= 191200):     ### change later 
                    current_date = datetime.now()
                    times_now = int(current_date.strftime("%H%M%S"))
                    if index_frame <= 150:
                        start_time = time.time() 						##start time of loop
                        num_frames = num_frames + 1
                        ret, frame = capture.read()
                        if not ret:
                            break

                        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight), swapRB=True, crop = False)
                        #get output layer
                        outNames = net.getUnconnectedOutLayersNames()
                        net.setInput(blob)
                        outs = net.forward(outNames)
                        current_detections, previous_frame_detections, vehicle_count, sr_no = process_frame(frame, outs, classes, CONF_THRESHOLD, NMS_THRESHOLD,
                            video_width, video_height, vehicle_count, num_frames, y_image, x_image, sr_no, scale, X_plot, Y_plot)
                        cv2.imshow('Press (q) to stop detections', frame)
                        out.write(frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        #update previous frame detection
                        previous_frame_detections.pop(0)
                        previous_frame_detections.append(current_detections)
                        index_frame = index_frame + 1
                        #print(index_frame, "ind froame no")
                        print(" inside loop", video_num, index_frame)
                    else:
                        clear()
                        video_num = video_num + 1
                        print(video_num, "video num")
                        out = cv2.VideoWriter('save_videos/'+str(video_num)+'.mp4', fourcc, 30, (960, 540))
                        index_frame = 0
                        cv2.imwrite('detection.jpg', frame)

                capture.release()
                out.release()
                cv2.destroyAllWindows()
                ######## output file name
                date_now = int(current_date.strftime("%Y%m%d"))
                print("saving data")
                wb.save("HELLO"+ '.xls')
                #wb.save(str(date_now) + '.xls')
                print("savED data")
            except:
                #shutil.rmtree('parameters.txt')
                print("error")
               
        daily_check = True


    else:
        print("not in time")
        if daily_check == True:
            print("generate sheets")
            ### loading file
            date_now = int(current_date.strftime("%Y%m%d"))
            daily_check =  sheet_generator("HELLO"+ '.xls')
            #daily_check = False #### done generating sheets
