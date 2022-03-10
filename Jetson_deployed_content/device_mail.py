from email import message
from itertools import count
import os
import socket
import struct
import cv2
import numpy
import base64
import sys
import time
import threading
from datetime import datetime
import pickle
from xlwt import Workbook
import csv
import numpy as np
import pytz
import smtplib
from email.message import EmailMessage
# import the necessary packages
import numpy as np
from scipy import spatial
#plotting the trajectory and bounding box
from matplotlib import image
from matplotlib import pyplot as plt
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
import shutil
import imutils
import struct
# define our clear function
def clear():
  
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

'''email parameters 
'''
email_session = smtplib.SMTP('smtp.gmail.com',587)
email_session.starttls()
email_session.login('autonomoustrafficanalysis@gmail.com', 'Auto@124')
msg = MIMEMultipart()
msg['Subject'] = 'Reports of Traffic'
msg['From'] = 'autonomoustrafficanalysis@gmail.com'
#msg['To'] = 'pareespathak@gmail.com'
mssg = MIMEMultipart()
mssg['Subject'] = 'Establishing connection on startup'
mssg['From'] = 'autonomoustrafficanalysis@gmail.com'
mssg['To'] = 'pareespathak@gmail.com'                        ##################input from user 
my_message = mssg.as_string()
email_session.sendmail('autonomoustrafficanalysis@gmail', 'pareespathak@gmail.com' ,my_message)
count_check = 0
video_num = 1
while True:
    try:
        Check = os.path.isfile('/home/parees/Downloads/Vehicle_tracking-main/Jetson/device/parameter.txt')
        time.sleep(2)
        clear()
        print("check =", Check)
        print("running main code")
        if Check == True:
            with open('/home/parees/Downloads/Vehicle_tracking-main/Jetson/device/parameter.txt') as f:
                src = eval(f.readline())
                dst = eval(f.readline())
                x_image = eval(f.readline())
                y_image = eval(f.readline())
                list_of_vehicles = eval(f.readline())
                if list_of_vehicles == None:
                    list_of_vehicles = ["car","bus","truck"]                                    ## Default classes
                email =f.readline()
                msg['To'] = email
                emailid = email
                print("email", emailid)

        current_date = datetime.now()
        times_now = int(current_date.strftime("%H%M%S"))
        ### change later
        #if  times_now <= 180000 and times_now >= 110000 and Check == True:
        try:
            if  times_now <= 114500 and times_now >= 100000 and Check == True:

                #check file name
                path_file = 'save_videos/'+ str(video_num)+'.mp4'
                Check = os.path.isfile(path_file)
                print("loop")
                if Check == True:
                    ##############  Size less than 20 mb
                    file = path_file
                    attachment = open(file,'rb')
                    obj = MIMEBase('application','octet-stream')
                    obj.set_payload((attachment).read())
                    encoders.encode_base64(obj)
                    obj.add_header('Content-Disposition',"attachment; filename= "+file)
                    msg.attach(obj)
                    my_message = msg.as_string()
                    print("sending mail")
                    email_session.sendmail('autonomoustrafficanalysis@gmail', emailid, my_message)
                    video_num = video_num + 1
                    msg = MIMEMultipart()
                    ## prepare to send
                    print(video_num)
                    try:
                        print("done sending video loop")
                        os.remove(file)
                        #video_num = video_num + 1
                    except Exception as e:
                        print("video_num cannot be deletted ", video_num + 1)
                        print(e)

                ## delete video file
        except:
            print("video cannot be send")

        ########### sending sheets
        length = os.listdir('/home/parees/Downloads/Vehicle_tracking-main/Jetson/device/sheets')
        size_of_folder = len(length)
        if times_now >= 123000 and times_now <= 240000 and size_of_folder >= 1:
            #file_content = os.listdir()
            file_size = 0 
            length = os.listdir('/home/parees/Downloads/Vehicle_tracking-main/Jetson/device/sheets')
            print(length)
            sr_no = 0
            for i in length:
                print(len(length))
                #save_name = 'sheets/generated_output' + str(i)+'.xls'
                print(i)
                save_name = '/home/parees/Downloads/Vehicle_tracking-main/Jetson/device/sheets/' + i 

                #if os.path.isfile(save_name) == True:
                #    file_size = file_size + (os.path.getsize(save_name))
                print(file_size)
                if file_size < 20*1000000:
                    print(type(save_name), "type")
                    sheet_attach = open(save_name, 'rb')
                    sheet_obj = MIMEBase('application','octet-stream')
                    sheet_obj.set_payload((sheet_attach).read())
                    encoders.encode_base64(sheet_obj)
                    sheet_obj.add_header('Content-Disposition',"attachment; filename= " + save_name)
                    msg.attach(sheet_obj)
                    sr_no = sr_no + 1
                    print("send 1")
                if file_size >= 20*1000000 or (len(length) == sr_no):
                    #print(len(length))
                    my_message = msg.as_string()
                    email_session.sendmail('autonomoustrafficanalysis@gmail', emailid, my_message)
                    msg = MIMEMultipart()
                    print("send of mail done")

            try:
                print("sending")
                shutil.rmtree('/home/parees/Downloads/Vehicle_tracking-main/Jetson/device/sheets')
                os.mkdir('/home/parees/Downloads/Vehicle_tracking-main/Jetson/device/sheets')
            except Exception as e:
                print(e)
                time.sleep(10)
            ### delete folder 
            ## recreate sheets folder for next day
    
    except Exception as e:
        print("error", e)
        time.sleep(600)
        count_check = count_check+1
        if count_check == (6*24):
            count_check = 0
            mssg = MIMEMultipart()
            mssg['Subject'] = 'Error in sending Email'
            mssg['From'] = 'autonomoustrafficanalysis@gmail.com'
            mssg['To'] = 'pareespathak@gmail.com'                        ##################input from user 
            my_message = mssg.as_string()
            email_session.sendmail('autonomoustrafficanalysis@gmail', 'pareespathak@gmail.com' ,my_message)
           


