# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt_designer.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
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

class Ui_MainWindow(object):
    i = 0 
    src_global, dst = [], []
    Gx_image, Gy_image = [], []
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(777, 502)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pathlabel = QtWidgets.QLabel(self.centralwidget)
        self.pathlabel.setGeometry(QtCore.QRect(10, 0, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pathlabel.setFont(font)
        self.pathlabel.setObjectName("pathlabel")
        self.videopath = QtWidgets.QLineEdit(self.centralwidget)
        self.videopath.setGeometry(QtCore.QRect(240, 10, 521, 22))
        self.videopath.setObjectName("videopath")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 100, 521, 351))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("../testing_video/scale_2.jpeg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 60, 121, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(500, 60, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.pointslabel = QtWidgets.QLineEdit(self.centralwidget)
        self.pointslabel.setGeometry(QtCore.QRect(640, 60, 41, 22))
        self.pointslabel.setObjectName("pointslabel")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(690, 60, 81, 21))
        self.label_4.setObjectName("label_4")
        self.detectionButton = QtWidgets.QPushButton(self.centralwidget)
        self.detectionButton.setGeometry(QtCore.QRect(200, 60, 271, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.detectionButton.setFont(font)
        self.detectionButton.setObjectName("detectionButton")
        self.p_1 = QtWidgets.QLabel(self.centralwidget)
        self.p_1.setGeometry(QtCore.QRect(560, 150, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.p_1.setFont(font)
        self.p_1.setObjectName("p_1")
        self.C_1x = QtWidgets.QLineEdit(self.centralwidget)
        self.C_1x.setGeometry(QtCore.QRect(660, 150, 31, 21))
        self.C_1x.setObjectName("C_1x")
        self.C_1y = QtWidgets.QLineEdit(self.centralwidget)
        self.C_1y.setGeometry(QtCore.QRect(710, 150, 31, 22))
        self.C_1y.setObjectName("C_1y")
        self.DoneButton = QtWidgets.QPushButton(self.centralwidget)
        self.DoneButton.setGeometry(QtCore.QRect(540, 410, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.DoneButton.setFont(font)
        self.DoneButton.setObjectName("DoneButton")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(670, 130, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(720, 130, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(550, 100, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(660, 100, 71, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(610, 190, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(550, 240, 41, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.pointnum = QtWidgets.QLabel(self.centralwidget)
        self.pointnum.setGeometry(QtCore.QRect(600, 240, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.pointnum.setFont(font)
        self.pointnum.setObjectName("pointnum")
        self.x_val = QtWidgets.QLabel(self.centralwidget)
        self.x_val.setGeometry(QtCore.QRect(660, 240, 41, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.x_val.setFont(font)
        self.x_val.setObjectName("x_val")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(710, 240, 41, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(550, 320, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.saveEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.saveEdit.setGeometry(QtCore.QRect(542, 340, 231, 22))
        self.saveEdit.setObjectName("saveEdit")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(620, 280, 81, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(670, 420, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(550, 380, 161, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 777, 26))
        self.menubar.setObjectName("menubar")
        self.menumenu = QtWidgets.QMenu(self.menubar)
        self.menumenu.setObjectName("menumenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menumenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        ###################################################################################4###############
        self.pushButton.clicked.connect(self.next_point)
        self.detectionButton.clicked.connect(self.detect_region)
        self.DoneButton.clicked.connect(self.done_action)
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pathlabel.setText(_translate("MainWindow", "Enter Video Path "))
        self.label_2.setText(_translate("MainWindow", "Sample View "))
        self.label_3.setText(_translate("MainWindow", "Enter Num of Points"))
        self.label_4.setText(_translate("MainWindow", "(Default = 4)"))
        self.detectionButton.setText(_translate("MainWindow", "Click Here To Mark The Trap Area"))
        self.p_1.setText(_translate("MainWindow", "Point"))
        self.DoneButton.setText(_translate("MainWindow", "Done"))
        self.label_5.setText(_translate("MainWindow", "X"))
        self.label_6.setText(_translate("MainWindow", "Y"))
        self.label_7.setText(_translate("MainWindow", " Image"))
        self.label_11.setText(_translate("MainWindow", "Real world"))
        self.pushButton.setText(_translate("MainWindow", "Next Point"))
        self.label_8.setText(_translate("MainWindow", "Point "))
        self.pointnum.setText(_translate("MainWindow", "1"))
        self.x_val.setText(_translate("MainWindow", "X val"))
        self.label_9.setText(_translate("MainWindow", "Y val"))
        self.label_10.setText(_translate("MainWindow", "Enter File Name"))
        self.label_12.setText(_translate("MainWindow", "Save File"))
        self.label_13.setText(_translate("MainWindow", "Status"))
        self.checkBox.setText(_translate("MainWindow", "Detect two wheelers"))
        self.menumenu.setTitle(_translate("MainWindow", "menu"))


    def done_action(self):
        try:
            path = self.videopath.text()
            videoStream = cv2.VideoCapture(path)
            ret, frame = videoStream.read()
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
            check = self.checkBox.isChecked()
            #print("chech", check)
            if check == True:
                list_of_vehicles = ["car","bus","truck", "train","bicycle","motorbike"]
            else:
                list_of_vehicles = ["car","bus","truck"]

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
                            if centerY >= np.min(y_image) and centerY <= np.max(y_image) and centerX >= np.min(x_image) and centerX <= np.max(x_image): 
                                color = [int(c) for c in COLORS[classIDs[i]]]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                    confidences[i])
                                cv2.putText(frame, text, (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                #text1 = "{}: {:.4f}".format(x,y)
                                #exporting data in sheet
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
                                sheet1.write(sr_no,8,float(h_real))
                                #if ID == 0:
                                #X_plot.append(centerX)
                                #Y_plot.append(centerY)
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

            # Read COCO dataset classes
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

            x_image, y_image = self.Gx_image, self.Gy_image
            X_plot, Y_plot = [], []
            video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #dst = [[0.0,0.0], [10.5, 0.0], [10.5, 30.0], [0.0, 30.0], [0.0, 20], [0.0, 10.0]]
            #x_image = [458, 734, 876, 241, 357, 423]
            #y_image = [186, 197, 578, 558, 358, 249]
            #src = [[458, 186], [734, 197], [876, 578], [241, 558], [357, 358], [423, 249]]
            homography_mat, Mask = cv2.findHomography(np.float32(self.src_global), np.float32(self.dst), method = cv2.RANSAC)
            #homography_mat, Mask = cv2.findHomography(np.float32(src), np.float32(dst), method = cv2.RANSAC)
            scale = homography_mat
            #Initialization
            previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]
            # previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees
            num_frames, vehicle_count, sr_no = 0, 0, 2
            _translate = QtCore.QCoreApplication.translate

            while( ret == True):
                start_time = time.time() 						##start time of loop
                num_frames = num_frames + 1
                #print(num_frames)
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
                cv2.imshow('Press (q) to stop detections', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                #update previous frame detection
                previous_frame_detections.pop(0)
                #print(type(current_detections))
                #previous_frame_detections.append(spatial.KDTree(current_detections))
                previous_frame_detections.append(current_detections)

                Fps = float(time.time() - start_time)           #Processing frame per seconds calculation
                #print("Fps = ", 1/Fps)
                Fps_string = "Fps = " + str(float(1/Fps))
                self.label_13.setText(_translate("MainWindow", Fps_string))


            cv2.destroyAllWindows()
            ######## output file name
            save_name = self.saveEdit.text()
            wb.save(save_name + '.xls')
            self.label_13.setText(_translate("MainWindow", "Saved File"))
            self.dst = []
            self.i = 1
        except:
            _translate = QtCore.QCoreApplication.translate
            self.label_13.setText(_translate("MainWindow", "Failed"))
            self.dst = []
            self.i = 1
            msg = QMessageBox()
            msg.setWindowTitle("Error occured")
            msg.setText("Detection error")
            x = msg.exec_()

    def next_point(self):
        try:
            if self.i <= int(self.pointslabel.text()):
                self.i = self.i+1
                _translate = QtCore.QCoreApplication.translate
                valuex = self.C_1x.text()
                valuey = self.C_1y.text()
                ###################################  try to detect error ####################
                #if valuex != None and valuey != None:
                self.C_1x.setText("")
                self.C_1y.setText("")
                x = valuex
                y = valuey
                #print(x,y,self.i)
                self.x_val.setText(valuex)
                self.label_9.setText(valuey)
                #self.pointnum.setText(self.i)
                self.dst.append([float(x),float(y)])
                self.pointnum.setText(_translate("MainWindow", str(self.i)))
            else:
                msg = QMessageBox()
                msg.setWindowTitle("Error occured")
                msg.setText("Num of points exceed")

                x = msg.exec_()

        
        except:
            msg = QMessageBox()
            msg.setWindowTitle("Error occured")
            msg.setText("Invalid Inputs")

            x = msg.exec_()


    def detect_region(self):
        try:
            src = []
            path = self.videopath.text()
            #'C:\aa\vehicle_tracking_college\tracking\yolo_youtube\datasets\delhi_validation.mp4'
            videoStream = cv2.VideoCapture(path)
            #drawing coordinates for image
            ret, frame = videoStream.read()
            
            image_ref = frame
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
            cv2.namedWindow("Double click to mark, press (q) after marking")
            cv2.setMouseCallback("Double click to mark, press (q) after marking",draw_coordinates)
            while (1):
                cv2.imshow('Double click to mark, press (q) after marking',image_ref)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.imwrite('detection.jpg',image_ref)
            cv2.destroyAllWindows()
            self.label.setPixmap(QtGui.QPixmap("detection.jpg"))

        except:
            #print("not detected")
            msg = QMessageBox()
            msg.setWindowTitle("Error occured")
            msg.setText("Video not detected")
            self.label_13.setText(_translate("MainWindow", "Failed"))
            self.dst = []
            self.i = 1

            x = msg.exec_()
        
        self.src_global = src              #### so that if someone dose mistake he can redraw again 
        self.Gx_image = x_image
        self.Gy_image = y_image


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
