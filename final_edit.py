# -*- coding: utf-8 -*-

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
    Path = ""
    src_global, dst = [], []
    Gx_image, Gy_image = [], []
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1000, 700)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 700))
        MainWindow.setMaximumSize(QtCore.QSize(1000, 700))
        MainWindow.setSizeIncrement(QtCore.QSize(4, 4))
        MainWindow.setBaseSize(QtCore.QSize(4, 4))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pathlabel = QtWidgets.QLabel(self.centralwidget)
        self.pathlabel.setGeometry(QtCore.QRect(10, 20, 401, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.pathlabel.setFont(font)
        self.pathlabel.setObjectName("pathlabel")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 260, 651, 391))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("detection_sample.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(250, 230, 121, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.detectionButton = QtWidgets.QPushButton(self.centralwidget)
        self.detectionButton.setGeometry(QtCore.QRect(460, 90, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.detectionButton.setFont(font)
        self.detectionButton.setObjectName("detectionButton")
        self.p_1 = QtWidgets.QLabel(self.centralwidget)
        self.p_1.setGeometry(QtCore.QRect(730, 130, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_1.setFont(font)
        self.p_1.setObjectName("p_1")
        self.DoneButton = QtWidgets.QPushButton(self.centralwidget)
        self.DoneButton.setGeometry(QtCore.QRect(900, 595, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.DoneButton.setFont(font)
        self.DoneButton.setObjectName("DoneButton")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(709, 100, 71, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.x_val = QtWidgets.QLabel(self.centralwidget)
        self.x_val.setGeometry(QtCore.QRect(830, 100, 21, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.x_val.setFont(font)
        self.x_val.setObjectName("x_val")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(900, 100, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(690, 490, 261, 61))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.saveEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.saveEdit.setGeometry(QtCore.QRect(690, 540, 231, 22))
        self.saveEdit.setObjectName("saveEdit")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(690, 570, 201, 35))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.checkBox_Lmv = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_Lmv.setGeometry(QtCore.QRect(20, 180, 71, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox_Lmv.setFont(font)
        self.checkBox_Lmv.setObjectName("checkBox_Lmv")
        self.Browse_but = QtWidgets.QPushButton(self.centralwidget)
        self.Browse_but.setGeometry(QtCore.QRect(420, 20, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Browse_but.setFont(font)
        self.Browse_but.setObjectName("Browse_but")
        self.checkBox_HMV = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_HMV.setGeometry(QtCore.QRect(20, 200, 61, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox_HMV.setFont(font)
        self.checkBox_HMV.setObjectName("checkBox_HMV")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(10, 55, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.Display_path = QtWidgets.QLabel(self.centralwidget)
        self.Display_path.setGeometry(QtCore.QRect(100, 55, 530, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.Display_path.setFont(font)
        self.Display_path.setObjectName("Display_path")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(10, 90, 441, 41))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(10, 140, 541, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.checkBox_Pedestrians = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_Pedestrians.setGeometry(QtCore.QRect(240, 180, 101, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox_Pedestrians.setFont(font)
        self.checkBox_Pedestrians.setObjectName("checkBox_Pedestrians")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(650, 10, 321, 91))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.checkBox_2W = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2W.setGeometry(QtCore.QRect(130, 180, 101, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox_2W.setFont(font)
        self.checkBox_2W.setObjectName("checkBox_2W")
        self.checkBox_Bicycle = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_Bicycle.setGeometry(QtCore.QRect(130, 200, 81, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox_Bicycle.setFont(font)
        self.checkBox_Bicycle.setObjectName("checkBox_Bicycle")
        self.X1 = QtWidgets.QLineEdit(self.centralwidget)
        self.X1.setGeometry(QtCore.QRect(810, 130, 51, 21))
        self.X1.setObjectName("X1")
        self.Y1 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y1.setGeometry(QtCore.QRect(880, 130, 51, 22))
        self.Y1.setObjectName("Y1")
        self.X2 = QtWidgets.QLineEdit(self.centralwidget)
        self.X2.setGeometry(QtCore.QRect(810, 160, 51, 21))
        self.X2.setObjectName("X2")
        self.Y2 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y2.setGeometry(QtCore.QRect(880, 160, 51, 22))
        self.Y2.setObjectName("Y2")
        self.p_2 = QtWidgets.QLabel(self.centralwidget)
        self.p_2.setGeometry(QtCore.QRect(730, 160, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_2.setFont(font)
        self.p_2.setObjectName("p_2")
        self.X3 = QtWidgets.QLineEdit(self.centralwidget)
        self.X3.setGeometry(QtCore.QRect(810, 190, 51, 21))
        self.X3.setObjectName("X3")
        self.Y3 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y3.setGeometry(QtCore.QRect(880, 190, 51, 22))
        self.Y3.setObjectName("Y3")
        self.p_3 = QtWidgets.QLabel(self.centralwidget)
        self.p_3.setGeometry(QtCore.QRect(730, 190, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_3.setFont(font)
        self.p_3.setObjectName("p_3")
        self.X4 = QtWidgets.QLineEdit(self.centralwidget)
        self.X4.setGeometry(QtCore.QRect(810, 220, 51, 21))
        self.X4.setObjectName("X4")
        self.Y4 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y4.setGeometry(QtCore.QRect(880, 220, 51, 22))
        self.Y4.setObjectName("Y4")
        self.p_4 = QtWidgets.QLabel(self.centralwidget)
        self.p_4.setGeometry(QtCore.QRect(730, 220, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_4.setFont(font)
        self.p_4.setObjectName("p_4")
        self.p_5 = QtWidgets.QLabel(self.centralwidget)
        self.p_5.setGeometry(QtCore.QRect(730, 310, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_5.setFont(font)
        self.p_5.setObjectName("p_5")
        ############################### X5
        self.X5 = QtWidgets.QLineEdit(self.centralwidget)
        self.X5.setGeometry(QtCore.QRect(810, 250, 51, 21))
        self.X5.setObjectName("X5")
        ###############################  Y5
        self.Y5 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y5.setGeometry(QtCore.QRect(880, 250, 51, 22))
        self.Y5.setObjectName("Y5")
        ###############################  X6
        self.X6 = QtWidgets.QLineEdit(self.centralwidget)
        self.X6.setGeometry(QtCore.QRect(810, 280, 51, 21))
        self.X6.setObjectName("X6")
        ###############################  Y6
        self.Y6 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y6.setGeometry(QtCore.QRect(880, 280, 51, 22))
        self.Y6.setObjectName("Y6")
        ###############################  X7
        self.X7 = QtWidgets.QLineEdit(self.centralwidget)
        self.X7.setGeometry(QtCore.QRect(810, 310, 51, 21))
        self.X7.setObjectName("X7")
        self.p_6 = QtWidgets.QLabel(self.centralwidget)
        self.p_6.setGeometry(QtCore.QRect(730, 340, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_6.setFont(font)
        self.p_6.setObjectName("p_6")
        ###############################  Y7
        self.Y7 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y7.setGeometry(QtCore.QRect(880, 310, 51, 22))
        self.Y7.setObjectName("Y7")
        ###############################  X8
        self.X8 = QtWidgets.QLineEdit(self.centralwidget)
        self.X8.setGeometry(QtCore.QRect(810, 340, 51, 21))
        self.X8.setObjectName("X8")
        self.p_7 = QtWidgets.QLabel(self.centralwidget)
        self.p_7.setGeometry(QtCore.QRect(730, 250, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_7.setFont(font)
        self.p_7.setObjectName("p_7")
        self.p_8 = QtWidgets.QLabel(self.centralwidget)
        self.p_8.setGeometry(QtCore.QRect(730, 280, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_8.setFont(font)
        self.p_8.setObjectName("p_8")
        ###############################  Y8
        self.Y8 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y8.setGeometry(QtCore.QRect(880, 340, 51, 22))
        self.Y8.setObjectName("Y8")
        ###############################  X9
        self.X9 = QtWidgets.QLineEdit(self.centralwidget)
        self.X9.setGeometry(QtCore.QRect(810, 370, 51, 21))
        self.X9.setObjectName("X9")
        ###############################  Y9
        self.Y9 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y9.setGeometry(QtCore.QRect(880, 370, 51, 22))
        self.Y9.setObjectName("Y9")
        self.X10 = QtWidgets.QLineEdit(self.centralwidget)
        self.X10.setGeometry(QtCore.QRect(810, 400, 51, 21))
        self.X10.setObjectName("X10")
        self.Y10 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y10.setGeometry(QtCore.QRect(880, 400, 51, 22))
        self.Y10.setObjectName("Y10")
        self.X11 = QtWidgets.QLineEdit(self.centralwidget)
        self.X11.setGeometry(QtCore.QRect(810, 430, 51, 21))
        self.X11.setObjectName("X11")
        self.Y11 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y11.setGeometry(QtCore.QRect(880, 430, 51, 22))
        self.Y11.setObjectName("Y11")
        self.X12 = QtWidgets.QLineEdit(self.centralwidget)
        self.X12.setGeometry(QtCore.QRect(810, 460, 51, 21))
        self.X12.setObjectName("X12")
        self.Y12 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y12.setGeometry(QtCore.QRect(880, 460, 51, 22))
        self.Y12.setObjectName("Y12")
        self.p_9 = QtWidgets.QLabel(self.centralwidget)
        self.p_9.setGeometry(QtCore.QRect(730, 430, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_9.setFont(font)
        self.p_9.setObjectName("p_9")
        self.p_10 = QtWidgets.QLabel(self.centralwidget)
        self.p_10.setGeometry(QtCore.QRect(730, 460, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_10.setFont(font)
        self.p_10.setObjectName("p_10")
        self.p_11 = QtWidgets.QLabel(self.centralwidget)
        self.p_11.setGeometry(QtCore.QRect(730, 370, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_11.setFont(font)
        self.p_11.setObjectName("p_11")
        self.p_12 = QtWidgets.QLabel(self.centralwidget)
        self.p_12.setGeometry(QtCore.QRect(730, 400, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_12.setFont(font)
        self.p_12.setObjectName("p_12")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(690, 610, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.Fps_display = QtWidgets.QLabel(self.centralwidget)
        self.Fps_display.setGeometry(QtCore.QRect(830, 610, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.Fps_display.setFont(font)
        self.Fps_display.setObjectName("Fps_display")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 26))
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
        self.Browse_but.clicked.connect(self.Browse_video)
        self.detectionButton.clicked.connect(self.detect_region)
        self.DoneButton.clicked.connect(self.done_action)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pathlabel.setText(_translate("MainWindow", "Click Browse to Select the Video for Traffic Detection"))
        self.label_2.setText(_translate("MainWindow", "Sample View "))
        self.detectionButton.setText(_translate("MainWindow", "Mark Trap Area"))
        self.p_1.setText(_translate("MainWindow", "Point 1"))
        self.DoneButton.setText(_translate("MainWindow", "RUN"))
        self.label_8.setText(_translate("MainWindow", "Point No."))
        self.x_val.setText(_translate("MainWindow", "X"))
        self.label_9.setText(_translate("MainWindow", "Y"))
        self.label_10.setText(_translate("MainWindow", "The Extracted Data will be Generated in an Excel \n"
        "Sheet. Enter the File Name Below:\n"
        ""))
        self.label_13.setText(_translate("MainWindow", "Processing speed depends on the \n"
        "processing power of the CPU & GPU. "))
        self.checkBox_Lmv.setText(_translate("MainWindow", "LMV"))
        self.Browse_but.setText(_translate("MainWindow", "Browse"))
        self.checkBox_HMV.setText(_translate("MainWindow", "HMV"))
        self.label_15.setText(_translate("MainWindow", "Selected File:"))
        self.Display_path.setText(_translate("MainWindow", "File Path"))
        self.label_17.setText(_translate("MainWindow", "Click the button to Mark the Trap Area for Traffic Detection\n"
        ""))
        self.label_18.setText(_translate("MainWindow", "Select the Type of Vehicles to be Detected & Extracted from the Video:\n"
        ""))
        self.checkBox_Pedestrians.setText(_translate("MainWindow", "Pedestrians"))
        self.label_19.setText(_translate("MainWindow", "Considering the Top Left Point of the Trap\n"
        "Area as Origin (0,0), enter the X & Y\n"
        "coordinates of All the Points Selected in\n"
        "the Trap Area:\n"
        ""))
        self.checkBox_2W.setText(_translate("MainWindow", "2 Wheelers"))
        self.checkBox_Bicycle.setText(_translate("MainWindow", "Bicycles"))
        self.p_2.setText(_translate("MainWindow", "Point 2"))
        self.p_3.setText(_translate("MainWindow", "Point 3"))
        self.p_4.setText(_translate("MainWindow", "Point 4"))
        self.p_5.setText(_translate("MainWindow", "Point 7"))
        self.p_6.setText(_translate("MainWindow", "Point 8"))
        self.p_7.setText(_translate("MainWindow", "Point 5"))
        self.p_8.setText(_translate("MainWindow", "Point 6"))
        self.p_9.setText(_translate("MainWindow", "Point 11"))
        self.p_10.setText(_translate("MainWindow", "Point 12"))
        self.p_11.setText(_translate("MainWindow", "Point 9"))
        self.p_12.setText(_translate("MainWindow", "Point 10"))
        self.label_3.setText(_translate("MainWindow", "Processing Speed:"))
        self.Fps_display.setText(_translate("MainWindow", "in %"))
        self.menumenu.setTitle(_translate("MainWindow", "menu"))

    def Browse_video(self):
        self.Path, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Single File', '', '*.mp4')
        _translate = QtCore.QCoreApplication.translate
        self.Display_path.setText(_translate("MainWindow", str(self.Path)))

        #########
    def detect_region(self):
        try:
            msg = QMessageBox()
            msg.setWindowTitle("Instructions For Marking")
            msg.setText(" 1. Double click to mark each point \n 2. Press (q) after all points are marked \n 3. Minimum 4 points are required. \n 4. Adding more points will increase accuracy \n 5. Maximum 12 points can be added")
            x = msg.exec_()
            src = []
            path = self.Path
            #'C:\aa\vehicle_tracking_college\tracking\yolo_youtube\datasets\delhi_validation.mp4'
            videoStream = cv2.VideoCapture(path)
            #drawing coordinates for image
            ret, frame = videoStream.read()
            image_ref = frame
            self.i = 0
            def draw_coordinates(event, x, y, flag, params):
                if event == cv2.EVENT_LBUTTONDBLCLK:

                    text = str(self.i+1)
                    cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    x_image.append(x)
                    y_image.append(y)
                    cv2.circle(image_ref, (x,y), 7, (0,0,255), -1)
                    src.append([x,y])
                    self.i = self.i + 1
                    if len(x_image) >= 2:
                        cv2.line(image_ref, (x_image[-1],y_image[-1]), (x_image[-2], y_image[-2]), (0, 200, 200), 3)

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
            x = msg.exec_()

        self.src_global = src              #### so that if someone dose mistake he can redraw again
        self.Gx_image = x_image
        self.Gy_image = y_image

    def done_action(self):
        try:
            path = self.Path
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
            #sheet1.write(0, 8, 'w_fac')
            #sheet1.write(0, 9, 'y_scale fac')
            sheet1.write(0, 8, 'Fps')
            sheet1.write(0, 11, 'time')
            # Define constants
            # CONF_THRESHOLD is confidence threshold. Only detection with confidence greater than this will be retained
            # NMS_THRESHOLD is used for non-max suppression
            CONF_THRESHOLD = 0.3
            #NMS to overcome the multiple boxes over single object
            NMS_THRESHOLD = 0.4
            list_of_vehicles = []
            if self.checkBox_Lmv.isChecked():
                list_of_vehicles.append("car")
            if self.checkBox_HMV.isChecked():
                list_of_vehicles.append("bus")
                list_of_vehicles.append("truck")
            if self.checkBox_2W.isChecked():
                list_of_vehicles.append("motorbike")
            if self.checkBox_Bicycle.isChecked():
                list_of_vehicles.append("bicycle")
            if self.checkBox_Pedestrians.isChecked():
                list_of_vehicles.append("person")
            if self.checkBox_Pedestrians.isChecked() == False and self.checkBox_Bicycle.isChecked() == False and self.checkBox_2W.isChecked() == False and self.checkBox_HMV.isChecked() == False and self.checkBox_Lmv.isChecked() == False:
                list_of_vehicles = ["car","bus","truck"]  ## Default classes
            #print(list_of_vehicles)
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
                                #sheet1.write(sr_no,8,float(h_real))
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
            Video_Fps = videoStream.get(cv2.CAP_PROP_FPS)
            sheet1.write(1, 8, Video_Fps)
            dst = []
            X_co = [self.X1.text(),self.X2.text(), self.X3.text(), self.X4.text(),
                    self.X5.text(),self.X6.text(), self.X7.text(), self.X8.text(),
                    self.X9.text(),self.X10.text(), self.X11.text(), self.X12.text()]
            Y_co = [self.Y1.text(),self.Y2.text(), self.Y3.text(), self.Y4.text(),
                    self.Y5.text(),self.Y6.text(), self.Y7.text(), self.Y8.text(),
                    self.Y9.text(),self.Y10.text(), self.Y11.text(), self.Y12.text()]
            try:
                for j in range(0,len(self.src_global)):
                    dst.append([float(X_co[j]),float(Y_co[j])])
            except:
                msg = QMessageBox()
                msg.setWindowTitle("Error occured")
                msg.setText("Invalid Inputs \n Inputs must be numbers")
                x = msg.exec_()

            if len(self.src_global) == len(dst):
                homography_mat, Mask = cv2.findHomography(np.float32(self.src_global), np.float32(dst), method = cv2.RANSAC)
                scale = homography_mat
            #homography_mat, Mask = cv2.findHomography(np.float32(src), np.float32(dst), method = cv2.RANSAC)
            else:
                msg = QMessageBox()
                msg.setWindowTitle("Error occured")
                msg.setText("Corresponding inputs number not matched")
                x = msg.exec_()
                ret = False
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
                Fps_string = float(1/Fps)
                Fps_string = str(int(Fps_string*100/Video_Fps)) + " %"
                self.Fps_display.setText(_translate("MainWindow", Fps_string))


            cv2.destroyAllWindows()
            ######## output file name
            save_name = self.saveEdit.text()
            wb.save(save_name + '.xls')
            self.Fps_display.setText(_translate("MainWindow", "Saved"))
            dst = []
        except:
            msg = QMessageBox()
            msg.setWindowTitle("Error occured")
            msg.setText("Failed To excecute")
            _translate = QtCore.QCoreApplication.translate
            self.Fps_display.setText(_translate("MainWindow", "Failed"))
            x = msg.exec_()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
