# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI_client.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from ast import While
from itertools import count
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
# import the necessary packages
import numpy as np
import numpy
import time
from scipy import spatial
import cv2
#from input_retrieval import *
from matplotlib import image
from matplotlib import pyplot as plt
import time
import socket
import base64
import glob
import sys
import threading
from datetime import datetime
import pickle
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import struct
import imutils

email_session = smtplib.SMTP('smtp.gmail.com',587)
email_session.starttls()
email_session.login('autonomoustrafficanalysis@gmail.com', 'Auto@124')

class Ui_MainWindow(object):
    i = 0 
    connectCount = 0
    TCP_SERVER_IP = ''
    TCP_SERVER_PORT = 0
    Path = ""
    src_global, dst_global = [], []
    Gx_image, Gy_image = [], []
    text_file_present = 0
    email_session = smtplib.SMTP('smtp.gmail.com',587)
    email_session.starttls()
    email_session.login('autonomoustrafficanalysis@gmail.com', 'Auto@124')
    sender_Email = ''
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 700)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 700))
        MainWindow.setMaximumSize(QtCore.QSize(1000, 700))
        MainWindow.setSizeIncrement(QtCore.QSize(4, 4))
        MainWindow.setBaseSize(QtCore.QSize(4, 4))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        ############X1
        self.X1 = QtWidgets.QLineEdit(self.centralwidget)
        self.X1.setGeometry(QtCore.QRect(820, 120, 51, 21))
        self.X1.setObjectName("X1")
        self.Y1 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y1.setGeometry(QtCore.QRect(890, 120, 51, 22))
        self.Y1.setObjectName("Y1")
        ###### 2
        self.X2 = QtWidgets.QLineEdit(self.centralwidget)
        self.X2.setGeometry(QtCore.QRect(820, 150, 51, 21))
        self.X2.setObjectName("X2")
        self.Y2 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y2.setGeometry(QtCore.QRect(890, 150, 51, 22))
        self.Y2.setObjectName("Y2")
        ###### 3
        self.X3 = QtWidgets.QLineEdit(self.centralwidget)
        self.X3.setGeometry(QtCore.QRect(820, 180, 51, 21))
        self.X3.setObjectName("X3")
        self.Y3 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y3.setGeometry(QtCore.QRect(890, 180, 51, 22))
        self.Y3.setObjectName("Y3")
        ### 4
        self.X4 = QtWidgets.QLineEdit(self.centralwidget)
        self.X4.setGeometry(QtCore.QRect(820, 210, 51, 21))
        self.X4.setObjectName("X4")
        self.Y4 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y4.setGeometry(QtCore.QRect(890, 210, 51, 22))
        self.Y4.setObjectName("Y4")
        ### 5 
        self.X5 = QtWidgets.QLineEdit(self.centralwidget)
        self.X5.setGeometry(QtCore.QRect(820, 240, 51, 21))
        self.X5.setObjectName("X5")
        self.Y5 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y5.setGeometry(QtCore.QRect(890, 240, 51, 22))
        self.Y5.setObjectName("Y5")
        ## 6 
        self.X6 = QtWidgets.QLineEdit(self.centralwidget)
        self.X6.setGeometry(QtCore.QRect(820, 270, 51, 21))
        self.X6.setObjectName("X6")
        self.Y6 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y6.setGeometry(QtCore.QRect(890, 270, 51, 22))
        self.Y6.setObjectName("Y6")
        ## 7 
        self.X7 = QtWidgets.QLineEdit(self.centralwidget)
        self.X7.setGeometry(QtCore.QRect(820, 300, 51, 21))
        self.X7.setObjectName("X7")
        self.Y7 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y7.setGeometry(QtCore.QRect(890, 300, 51, 22))
        self.Y7.setObjectName("Y7")
        ## 8
        self.X8 = QtWidgets.QLineEdit(self.centralwidget)
        self.X8.setGeometry(QtCore.QRect(820, 330, 51, 21))
        self.X8.setObjectName("X8")
        self.Y8 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y8.setGeometry(QtCore.QRect(890, 330, 51, 22))
        self.Y8.setObjectName("Y8")
        ## 9
        self.X9 = QtWidgets.QLineEdit(self.centralwidget)
        self.X9.setGeometry(QtCore.QRect(820, 360, 51, 21))
        self.X9.setObjectName("X9")
        self.Y9 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y9.setGeometry(QtCore.QRect(890, 360, 51, 22))
        self.Y9.setObjectName("Y9")
        ## 10
        self.X10 = QtWidgets.QLineEdit(self.centralwidget)
        self.X10.setGeometry(QtCore.QRect(820, 390, 51, 21))
        self.X10.setObjectName("X10")
        self.Y10 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y10.setGeometry(QtCore.QRect(890, 390, 51, 22))
        self.Y10.setObjectName("Y10")
        ## 11
        self.X11 = QtWidgets.QLineEdit(self.centralwidget)
        self.X11.setGeometry(QtCore.QRect(820, 420, 51, 21))
        self.X11.setObjectName("X11")
        self.Y11 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y11.setGeometry(QtCore.QRect(890, 420, 51, 22))
        self.Y11.setObjectName("Y11")
        ## 12 
        self.X12 = QtWidgets.QLineEdit(self.centralwidget)
        self.X12.setGeometry(QtCore.QRect(820, 450, 51, 21))
        self.X12.setObjectName("X12")
        self.Y12 = QtWidgets.QLineEdit(self.centralwidget)
        self.Y12.setGeometry(QtCore.QRect(890, 450, 51, 22))
        self.Y12.setObjectName("Y12")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(260, 220, 121, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(690, 480, 301, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.p_2 = QtWidgets.QLabel(self.centralwidget)
        self.p_2.setGeometry(QtCore.QRect(740, 150, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_2.setFont(font)
        self.p_2.setObjectName("p_2")
        self.DoneButton = QtWidgets.QPushButton(self.centralwidget)
        self.DoneButton.setGeometry(QtCore.QRect(880, 600, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.DoneButton.setFont(font)
        self.DoneButton.setObjectName("DoneButton")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(20, 150, 541, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(719, 90, 71, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.pathlabel = QtWidgets.QLabel(self.centralwidget)
        self.pathlabel.setGeometry(QtCore.QRect(20, 10, 461, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.pathlabel.setFont(font)
        self.pathlabel.setObjectName("pathlabel")
        self.p_5 = QtWidgets.QLabel(self.centralwidget)
        self.p_5.setGeometry(QtCore.QRect(740, 300, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_5.setFont(font)
        self.p_5.setObjectName("p_5")
        self.p_12 = QtWidgets.QLabel(self.centralwidget)
        self.p_12.setGeometry(QtCore.QRect(740, 390, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_12.setFont(font)
        self.p_12.setObjectName("p_12")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(910, 90, 16, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 250, 651, 391))
        self.label.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.label.setMouseTracking(False)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("../../../testing_video/scale_2.jpeg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.Browse_but = QtWidgets.QPushButton(self.centralwidget)
        self.Browse_but.setGeometry(QtCore.QRect(500, 10, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Browse_but.setFont(font)
        self.Browse_but.setObjectName("Browse_but")
        self.p_1 = QtWidgets.QLabel(self.centralwidget)
        self.p_1.setGeometry(QtCore.QRect(740, 120, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_1.setFont(font)
        self.p_1.setObjectName("p_1")
        self.p_3 = QtWidgets.QLabel(self.centralwidget)
        self.p_3.setGeometry(QtCore.QRect(740, 180, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_3.setFont(font)
        self.p_3.setObjectName("p_3")
        self.p_9 = QtWidgets.QLabel(self.centralwidget)
        self.p_9.setGeometry(QtCore.QRect(740, 420, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_9.setFont(font)
        self.p_9.setObjectName("p_9")
        self.p_7 = QtWidgets.QLabel(self.centralwidget)
        self.p_7.setGeometry(QtCore.QRect(740, 240, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_7.setFont(font)
        self.p_7.setObjectName("p_7")
        self.checkBox_HMV = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_HMV.setGeometry(QtCore.QRect(470, 180, 61, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox_HMV.setFont(font)
        self.checkBox_HMV.setObjectName("checkBox_HMV")
        self.p_6 = QtWidgets.QLabel(self.centralwidget)
        self.p_6.setGeometry(QtCore.QRect(740, 330, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_6.setFont(font)
        self.p_6.setObjectName("p_6")
        self.checkBox_2W = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2W.setGeometry(QtCore.QRect(140, 180, 101, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox_2W.setFont(font)
        self.checkBox_2W.setObjectName("checkBox_2W")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(20, 80, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.checkBox_Pedestrians = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_Pedestrians.setGeometry(QtCore.QRect(250, 180, 101, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox_Pedestrians.setFont(font)
        self.checkBox_Pedestrians.setObjectName("checkBox_Pedestrians")
        self.x_val = QtWidgets.QLabel(self.centralwidget)
        self.x_val.setGeometry(QtCore.QRect(840, 90, 21, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.x_val.setFont(font)
        self.x_val.setObjectName("x_val")
        self.p_10 = QtWidgets.QLabel(self.centralwidget)
        self.p_10.setGeometry(QtCore.QRect(740, 450, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_10.setFont(font)
        self.p_10.setObjectName("p_10")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(690, 550, 161, 71))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.saveEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.saveEdit.setGeometry(QtCore.QRect(690, 520, 291, 22))
        self.saveEdit.setObjectName("saveEdit")
        self.p_11 = QtWidgets.QLabel(self.centralwidget)
        self.p_11.setGeometry(QtCore.QRect(740, 360, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_11.setFont(font)
        self.p_11.setObjectName("p_11")
        self.p_4 = QtWidgets.QLabel(self.centralwidget)
        self.p_4.setGeometry(QtCore.QRect(740, 210, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_4.setFont(font)
        self.p_4.setObjectName("p_4")
        self.checkBox_Bicycle = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_Bicycle.setGeometry(QtCore.QRect(370, 180, 81, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox_Bicycle.setFont(font)
        self.checkBox_Bicycle.setObjectName("checkBox_Bicycle")
        self.p_8 = QtWidgets.QLabel(self.centralwidget)
        self.p_8.setGeometry(QtCore.QRect(740, 270, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.p_8.setFont(font)
        self.p_8.setObjectName("p_8")
        self.checkBox_Lmv = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_Lmv.setGeometry(QtCore.QRect(30, 180, 71, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.checkBox_Lmv.setFont(font)
        self.checkBox_Lmv.setObjectName("checkBox_Lmv")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(670, 0, 331, 91))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.Display_path = QtWidgets.QLabel(self.centralwidget)
        self.Display_path.setGeometry(QtCore.QRect(120, 80, 301, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.Display_path.setFont(font)
        self.Display_path.setObjectName("Display_path")
        self.X1_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.X1_2.setGeometry(QtCore.QRect(110, 50, 231, 21))
        self.X1_2.setObjectName("X1_2")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(20, 50, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(360, 50, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.X1_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.X1_3.setGeometry(QtCore.QRect(500, 50, 101, 21))
        self.X1_3.setObjectName("X1_3")
        self.Browse_but_2 = QtWidgets.QPushButton(self.centralwidget)
        self.Browse_but_2.setGeometry(QtCore.QRect(430, 110, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Browse_but_2.setFont(font)
        self.Browse_but_2.setObjectName("Browse_but_2")
        self.Browse_but_3 = QtWidgets.QPushButton(self.centralwidget)
        self.Browse_but_3.setGeometry(QtCore.QRect(270, 110, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Browse_but_3.setFont(font)
        self.Browse_but_3.setObjectName("Browse_but_3")
        self.DoneButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.DoneButton_2.setGeometry(QtCore.QRect(860, 550, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.DoneButton_2.setFont(font)
        self.DoneButton_2.setObjectName("DoneButton_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        #############################################################################
        self.Browse_but_2.clicked.connect(self.detect_region)  ######change button name
        self.Browse_but.clicked.connect(self.Connect)  ######change button name
        self.Browse_but_3.clicked.connect(self.Previous_data)
        self.DoneButton.clicked.connect(self.Send_data)
        self.DoneButton_2.clicked.connect(self.Check_Email)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "Sample View "))
        self.label_10.setText(_translate("MainWindow", "Type Email Address (Gmail) below to recieve \n"
        "extracted data and video file and click Confirm Email"))
        self.p_2.setText(_translate("MainWindow", "Point 2"))
        self.DoneButton.setText(_translate("MainWindow", "RUN"))
        self.label_18.setText(_translate("MainWindow", "Select the Type of Vehicles to be Detected & Extracted from the Video:"))
        self.label_8.setText(_translate("MainWindow", "Point No."))
        self.pathlabel.setText(_translate("MainWindow", "Type IP Address and Password of the Server then click connect"))
        self.p_5.setText(_translate("MainWindow", "Point 7"))
        self.p_12.setText(_translate("MainWindow", "Point 10"))
        self.label_9.setText(_translate("MainWindow", "Y"))
        self.Browse_but.setText(_translate("MainWindow", "Connect"))
        self.p_1.setText(_translate("MainWindow", "Point 1"))
        self.p_3.setText(_translate("MainWindow", "Point 3"))
        self.p_9.setText(_translate("MainWindow", "Point 11"))
        self.p_7.setText(_translate("MainWindow", "Point 5"))
        self.checkBox_HMV.setText(_translate("MainWindow", "HMV"))
        self.p_6.setText(_translate("MainWindow", "Point 8"))
        self.checkBox_2W.setText(_translate("MainWindow", "2 Wheelers"))
        self.label_15.setText(_translate("MainWindow", "Connected to:"))
        self.checkBox_Pedestrians.setText(_translate("MainWindow", "Pedestrians"))
        self.x_val.setText(_translate("MainWindow", "X"))
        self.p_10.setText(_translate("MainWindow", "Point 12"))
        self.label_13.setText(_translate("MainWindow", "Confirmation email has been \n"
        "send to the above address. \n"
        "Click Run to proceed else \n"
        "Check email address"))
        self.p_11.setText(_translate("MainWindow", "Point 9"))
        self.p_4.setText(_translate("MainWindow", "Point 4"))
        self.checkBox_Bicycle.setText(_translate("MainWindow", "Bicycles"))
        self.p_8.setText(_translate("MainWindow", "Point 6"))
        self.checkBox_Lmv.setText(_translate("MainWindow", "LMV"))
        self.label_19.setText(_translate("MainWindow", "Considering the Top Left Point of the Trap\n"
        "Area as Origin (0,0), enter the X & Y\n"
        "coordinates of All the Points Selected in\n"
        "the Trap Area:"))
        self.Display_path.setText(_translate("MainWindow", "Address"))
        self.label_16.setText(_translate("MainWindow", "IP Address "))
        self.label_17.setText(_translate("MainWindow", "Port No."))
        self.Browse_but_2.setText(_translate("MainWindow", "Start New Region"))
        self.Browse_but_3.setText(_translate("MainWindow", "Run Previous"))
        self.DoneButton_2.setText(_translate("MainWindow", "Confirm Email"))
    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf
    
    def detect_region(self):
        try:
            msg = QMessageBox()
            msg.setWindowTitle("Instructions For Marking")
            msg.setText(" 1. Double click to mark each point \n 2. Press (q) after all points are marked \n 3. Minimum 4 points are required. \n 4. Adding more points will increase accuracy \n 5. Maximum 12 points can be added")
            x = msg.exec_()
            src = []
            x_image, y_image = [], []
            ###########################################################################
            self.sock.send("1".encode('utf-8'))
            Request = True
            permission = None
            count = 0
            try:
                print("recieving image")
                data = b""
                payload_size = struct.calcsize("Q")
                while len(data) < payload_size:
                    packet = self.sock.recv(4*1024)
                    if not packet: break
                    data+=packet
                packet_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packet_msg_size)[0]
                
                while len(data) < msg_size:
                    data += self.sock.recv(4*1024)
                frame_data = data[:msg_size]
                data = data[msg_size:]
                frame = pickle.loads(frame_data)
                #cv2.imshow('Double click to mark, press (q) after marking', frame)
                frame = imutils.resize(frame, width = 640)
                print(count)
                count = count+1
            except socket.timeout as e:
                err = e.args[0]
                # this next if/else is a bit redundant, but illustrates how the
                # timeout exception is setup
                if err == 'timed out':
                    time.sleep(1)
                    print('recv timed out, retry later')
                else:
                    print (e)
                    sys.exit(1)
            except socket.error as e:
                # Something else happened, handle error, exit, etc.
                print (e)
                #sys.exit(1)
            else:
                if len(data) == 0:
                    print ('y 0')
                    #sys.exit(0)


            '''
            while Request:
                length = self.recvall(self.sock, 64)
                length1 = length.decode('utf-8')
                print("length", length1)
                stringData = self.recvall(self.sock, int(length1))
                data = numpy.frombuffer(base64.b64decode(stringData), numpy.uint8)
                decimg = cv2.imdecode(data, 1)
                image_ref = decimg
                Request = False
            '''
            #drawing coordinates for image
            print("recieved image")
            #frame = image_ref
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
            
        except Exception as e:
            print(e)
            #print("not detected")
            msg = QMessageBox()
            msg.setWindowTitle("Error occured")
            msg.setText("Video not detected")
            x = msg.exec_()

        self.src_global = src              #### so that if someone dose mistake he can redraw again
        self.Gx_image = x_image
        self.Gy_image = y_image

    def Previous_data(self):
        print("sending req to check text file")
        self.sock.send("2".encode('utf-8'))
        ##############################################
        file_available = self.sock.recv(1024).decode('utf-8')

        if file_available == "a":
            msg = QMessageBox()
            msg.setWindowTitle("Previous info")
            msg.setText("File found, complete the process is restarted")
            x = msg.exec_()
            sys.exit()                                                      ########### ending connections 
        if file_available == "b":
            msg = QMessageBox()
            msg.setWindowTitle("Previous info")
            msg.setText("File not found, Reconnnect and click on new connection")
            x = msg.exec_()
            #self.sock.close()
            self.text_file_present = 1

    def Check_Email(self):
        try:
            Senders_gmail = self.saveEdit.text()
            mssg = MIMEMultipart()
            mssg['Subject'] = 'Verification of Email'
            mssg['From'] = 'autonomoustrafficanalysis@gmail.com'
            mssg['To'] = Senders_gmail                        ##################input from user 
            my_message = mssg.as_string()
            self.email_session.sendmail('autonomoustrafficanalysis@gmail', Senders_gmail ,my_message)
            msg = QMessageBox()
            msg.setWindowTitle("Email Verified")
            msg.setText("Verification email has been send to the provided email. \n Please check the spam folder ")
            x = msg.exec_()
            self.sender_Email = Senders_gmail

        except:
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Email Verification cannot be completed. \n Please provide valid GMAIL address. \n or reconnect the client")
            x = msg.exec_()

        
        pass

    def Send_data(self):
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
        if self.checkBox_Pedestrians.isChecked() == False or self.checkBox_Bicycle.isChecked() == False or self.checkBox_2W.isChecked() == False or self.checkBox_HMV.isChecked() == False or self.checkBox_Lmv.isChecked() == False:
            list_of_vehicles = ["car","bus","truck"]  ## Default classes
        dst = []
        ########## take data of coorinates from boxes 
        X_co = [self.X1.text(),self.X2.text(), self.X3.text(), self.X4.text(),
                self.X5.text(),self.X6.text(), self.X7.text(), self.X8.text(),
                self.X9.text(),self.X10.text(), self.X11.text(), self.X12.text()]
        Y_co = [self.Y1.text(),self.Y2.text(), self.Y3.text(), self.Y4.text(),
                self.Y5.text(),self.Y6.text(), self.Y7.text(), self.Y8.text(),
                self.Y9.text(),self.Y10.text(), self.Y11.text(), self.Y12.text()]
        try:
            for j in range(0,len(self.src_global)):
                dst.append([float(X_co[j]),float(Y_co[j])])
                #if self.text_file_present == 1:

            ############ sending src and dst
            src = pickle.dumps(self.src_global)
            self.sock.sendall(src)
            time.sleep(1)
            #print("senfing dst")
            dst = pickle.dumps(dst)
            self.sock.sendall(dst)
            time.sleep(1)
            print("Gy to send",self.Gy_image)
            x_image = pickle.dumps(self.Gx_image)
            self.sock.sendall(x_image)
            ####### sending IP and Email
            time.sleep(1)
            y_image = pickle.dumps(self.Gy_image)
            self.sock.sendall(y_image)
            time.sleep(1)
            #print("senfing email")
            Email = self.saveEdit.text()            #get from command line
            print("email",Email)
            Email = pickle.dumps(Email)
            self.sock.sendall(Email)
            time.sleep(1)
            ############
            List = pickle.dumps(list_of_vehicles)
            self.sock.sendall(List)
            time.sleep(10)
            
            msg = QMessageBox()
            msg.setWindowTitle("Data send")
            msg.setText("Data has been send. \n Please close the application.")
            x = msg.exec_()
            #self.sock.close()

            #self.sock.close()
            #sys.exit()                                                      ########### ending connections 

        except:
            msg = QMessageBox()
            msg.setWindowTitle("Error occured")
            msg.setText("Invalid Inputs \n Inputs must be numbers")
            x = msg.exec_()

        '''
        if self.text_file_present == 0:
            print("Working on previous data")
            task = pickle.dumps("y")
            self.sock.send(task)
        '''
        
        
     
    def Connect(self):
        #TCP_IP = self.X1_2.text()                        ##########     ip from text box
        
        #self.TCP_SERVER_IP = float(self.X1_2.text())
        #self.TCP_SERVER_PORT = 8080                                  ##########   fix port number of all Jetson 
        #Password = float(self.X1_2.text())               ##########    get password instead of port number
        #client = ClientSocket(TCP_IP, TCP_PORT)
        self.TCP_SERVER_PORT = 8080
        '''
        try: 
            Password = float(self.X1_2.text())    
        except:
            #print("not detected")
            msg = QMessageBox()
            msg.setWindowTitle("Error occured")
            msg.setText("Password in correct")
            x = msg.exec_()
        '''
        try: 
            #192.168.29.176
            #socket.AF_INET6
            self.TCP_SERVER_IP = str(self.X1_2.text())
            print(self.TCP_SERVER_IP, type(self.TCP_SERVER_IP))
            #self.TCP_SERVER_IP = 'localhost' 
            self.TCP_SERVER_PORT = 5005
            #self.TCP_SERVER_IP = 'localhost' 
        except Exception as e:
            print(e,"not detected")
            msg = QMessageBox()
            msg.setWindowTitle("Error occured")
            msg.setText("Ip address inncorrect")
            x = msg.exec_()
        try:
            self.sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            self.sock.connect((self.TCP_SERVER_IP, self.TCP_SERVER_PORT))
            print(u'Client socket is connected with Server socket [ TCP_SERVER_IP: ' + self.TCP_SERVER_IP + ', TCP_SERVER_PORT: ' + str(self.TCP_SERVER_PORT) + ' ]')
            self.connectCount = 0
            #self.sendImages()
        except Exception as e:
            print(e)
            self.connectCount += 1
            if self.connectCount == 3:
                print(u'Connect fail %d times. exit program'%(self.connectCount))
                sys.exit()
            print(u'%d times try to connect with server'%(self.connectCount))
            msg = QMessageBox()
            msg.setWindowTitle("Connection Error occured")
            msg.setText("Reasons \n 1. Check the connection of server \n 2. Check IP address \n 3. Select Port number to 8080 \n 4. Switch off previous clients")
            x = msg.exec_()
            self.Connect()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
