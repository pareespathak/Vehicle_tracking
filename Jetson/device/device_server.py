import os
import socket
from tabnanny import check
import cv2
import numpy
import base64
import glob
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
from email.mime.multipart import MIMEMultipart
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


'''email parameters 
'''
email_session = smtplib.SMTP('smtp.gmail.com',587)
email_session.starttls()
email_session.login('autonomoustrafficanalysis@gmail.com', 'Auto@124')
msg = MIMEMultipart()
msg['Subject'] = 'Hello'
msg['From'] = 'autonomoustrafficanalysis@gmail.com'
#msg['To'] = 'pareespathak@gmail.com'

'''initially setting up socket 
'''
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
Socket_IP = 'localhost'
Socket_Port = 8080
## bind 
server_socket.bind((Socket_IP, Socket_Port))
### listen 
server_socket.listen(1)
print("listning to client for first input")
#conn, address = server_socket.accept()
server_socket.settimeout(10)

def sendImages():
    try:
      frame = cv2.imread("detection.jpg", cv2.IMREAD_COLOR)
    except:
      capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)             ######change dshow parameter as per linux 
      ret, frame = capture.read()
    stime = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = base64.b64encode(data)
    length = str(len(stringData))
    conn.sendall(length.encode('utf-8').ljust(64))
    conn.send(stringData)
    conn.send(stime.encode('utf-8').ljust(64))
    time.sleep(1)
    cv2.destroyAllWindows 
    print("image send from server")
    #return ret
    print("replace text file")
    ######### src and dst
    src = conn.recv(2048)
    src = pickle.loads(src)
    print("src",src)
    dst = conn.recv(2048)
    dst = pickle.loads(dst)
    ############# x_image , y_image
    x_image = conn.recv(2048)
    x_image = pickle.loads(x_image)
    y_image = conn.recv(2048)
    y_image = pickle.loads(y_image)
    ######## email and list of veh
    email = conn.recv(2048)
    email = pickle.loads(email)
    list_of_vehicles = conn.recv(2048)
    list_of_vehicles = pickle.loads(list_of_vehicles)
    print("email", email, "dst", dst)
    #file = open("", "w+")
    with open('parameters.txt', 'w') as f:
        f.write(str(src))
        f.write('\n')
        f.write(str(dst))
        f.write('\n')
        f.write(str(x_image))
        f.write('\n')
        f.write(str(y_image))
        f.write('\n')
        f.write(str(list_of_vehicles))
        f.write('\n')
        f.write(str(email))

conn = False

file_size = os.path.getsize('d:/file.jpg') 
while True:
    video_num = 0
    Text_file = 10
    try:
        conn, address = server_socket.accept()
        print("conn", conn)
        if conn:
            print("recieved connection from", address)
            task = conn.recv(1024).decode('utf-8')
            print(task)
            if not task:
                print("nahi")
                Text_file = 10
            elif task == "1":
                print("sending image")
                receiveThread = threading.Thread(target=sendImages)
                receiveThread.start()
                print("image send from server")
                Text_file = 12
                pass
            
            elif task == "2":
                ######### previous data 
                print("checking text file")
                path_file = 'parameters.txt'
                #print(path_file)
                Check = os.path.isfile(path_file)
                print("check =", Check)
                #print("file exists")
                conn.send("a".encode('utf-8'))
                #send text of file found
                #Text_file = 10
                if Check == False:
                  print("request file")
                  #send text file not found, crete filr 
                  conn.send("b".encode('utf-8'))
                  receiveThread = threading.Thread(target=sendImages)
                  receiveThread.start()
                  print("image send from server")
                  #Text_file = 12
                              
            else:
                print("extract data from current file")
                print(task)

    except socket.timeout:
        print("timeout error")
    
    
    current_date = datetime.now()
    times_now = int(current_date.strftime("%H%M%S"))
    if  times_now <= 240000 and times_now >= 200000:

      Check = os.path.isfile(path_file)
      print("check =", Check)
      print("running main code")
      if Check == True:
         with open(path_file) as f:
               src = eval(f.readline())
               dst = eval(f.readline())
               x_image = eval(f.readline())
               y_image = eval(f.readline())
               list_of_vehicles = eval(f.readline())
               if list_of_vehicles == None:
                  list_of_vehicles = ["car","bus","truck"]                                    ## Default classes
               email =f.readline()
               msg['To'] = email


         ######## sending email
         
         name_id = 1
         while file_size <= 20:
            #attacch

         save_name = 'sheets/generated output' + str(name_id) + '.xls'
         file_size = os.stat(save_name)

         sheet1 = save_name + '.xls'
         sheet_attach = open(sheet1, 'rb')
         sheet_obj = MIMEBase('application','octet-stream')
         sheet_obj.set_payload((sheet_attach).read())
         encoders.encode_base64(sheet_obj)
         sheet_obj.add_header('Content-Disposition',"attachment; filename= "+sheet1)
         msg.attach(sheet_obj)
         my_message = msg.as_string()
         email_session.sendmail('autonomoustrafficanalysis@gmail', email, my_message)
         print("email send")


        
    print("end of connection ")    
    print("inputs provoded = \n")    
    print(src,dst,list_of_vehicles)





conn = False

file_size = os.path.getsize('d:/file.jpg') 
while True:
    try:
        conn, address = server_socket.accept()
        print("conn", conn)
        if conn:
            print("recieved connection from", address)
            task = conn.recv(1024).decode('utf-8')
            print(task)
            if not task:
                print("nahi")
                Text_file = 10
            elif task == "1":
                print("sending image")
                receiveThread = threading.Thread(target=sendImages)
                receiveThread.start()
                print("image send from server")
                pass
            elif task == "2":
                ######### previous data 
                print("checking text file")
                path_file = 'parameters.txt'
                #print(path_file)
                Check = os.path.isfile(path_file)
                print("check =", Check)
                #print("file exists")
                conn.send("a".encode('utf-8'))
                #send text of file found
                #Text_file = 10
                if Check == False:
                  print("request file")
                  #send text file not found, crete filr 
                  conn.send("b".encode('utf-8'))
                  receiveThread = threading.Thread(target=sendImages)
                  receiveThread.start()
                  print("image send from server")
                  #Text_file = 12               
            else:
                print("extract data from current file")
                print(task)

    except socket.timeout:
      print("timeout error")
    
    
    current_date = datetime.now()
    times_now = int(current_date.strftime("%H%M%S"))
    if times_now >= 80000 and times_now <= 190000:
      print("sheet sending off")
      print("video sending on")
      ##############  Size less than 20 mb
      file_size = os.path.getsize('d:/file.jpg')
      sending_size = 0
      while sending_size < 20:

      file = "output.mp4"
      attachment = open(file,'rb')
      obj = MIMEBase('application','octet-stream')
      obj.set_payload((attachment).read())
      encoders.encode_base64(obj)
      obj.add_header('Content-Disposition',"attachment; filename= "+file)
      msg.attach(obj)

      sheet1 = "testing.xls"
      sheet_attach = open(sheet1, 'rb')
      sheet_obj = MIMEBase('application','octet-stream')
      sheet_obj.set_payload((sheet_attach).read())
      encoders.encode_base64(sheet_obj)
      sheet_obj.add_header('Content-Disposition',"attachment; filename= "+sheet1)
      msg.attach(sheet_obj)


      my_message = msg.as_string()

      email_session.sendmail('autonomoustrafficanalysis@gmail', 'pareespathak@gmail.com',my_message)

    if  times_now <= 240000 and times_now >= 200000:
       

      Check = os.path.isfile(path_file)
      print("check =", Check)
      print("running main code")
      if Check == True:
         with open(path_file) as f:
               src = eval(f.readline())
               dst = eval(f.readline())
               x_image = eval(f.readline())
               y_image = eval(f.readline())
               list_of_vehicles = eval(f.readline())
               if list_of_vehicles == None:
                  list_of_vehicles = ["car","bus","truck"]                                    ## Default classes
               email =f.readline()
               msg['To'] = email


         ######## sending email
         
         name_id = 1
         while file_size <= 20:
            #attacch
            
         save_name = 'sheets/generated output' + str(name_id) + '.xls'
         file_size = os.stat(save_name)

         sheet1 = save_name + '.xls'
         sheet_attach = open(sheet1, 'rb')
         sheet_obj = MIMEBase('application','octet-stream')
         sheet_obj.set_payload((sheet_attach).read())
         encoders.encode_base64(sheet_obj)
         sheet_obj.add_header('Content-Disposition',"attachment; filename= "+sheet1)
         msg.attach(sheet_obj)
         my_message = msg.as_string()
         email_session.sendmail('autonomoustrafficanalysis@gmail', email, my_message)
         print("email send")


        
    print("end of connection ")    
    print("inputs provoded = \n")    
    print(src,dst,list_of_vehicles)