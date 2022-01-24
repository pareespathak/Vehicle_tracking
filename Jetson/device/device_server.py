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
# import the necessary packages
import numpy as np
from scipy import spatial
#plotting the trajectory and bounding box
from matplotlib import image
from matplotlib import pyplot as plt
import shutil
import imutils
import struct

'''initially setting up socket 
'''
server_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
Host_name = socket.gethostname()
Host_IP = socket.gethostbyname(Host_name)
print("name", Host_name, "ip", Host_IP)
Socket_IP = 'localhost'
Socket_IP = '2405:201:1003:f00a:3449:452b:7dc7:8bd2'
#Socket_IP = '2402:8100:2453:4f66:3449:452b:7dc7:8bd2'
Socket_Port = 8080
## bind 
server_socket.bind((Socket_IP, Socket_Port))
### listen 
server_socket.listen(1)
print("listning to client for first input")
#conn, address = server_socket.accept()
#server_socket.settimeout(10)
def recvall(sock, count):
  buf = b''
  while count:
      newbuf = sock.recv(count)
      if not newbuf: return None
      buf += newbuf
      count -= len(newbuf)
  return buf
def sendImages(conn):
    try:
      frame = cv2.imread("detection.jpg", cv2.IMREAD_COLOR)
    except:
      capture = cv2.VideoCapture(0)             ######change dshow parameter as per linux 
      ret, frame = capture.read()
    '''
    stime = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = base64.b64encode(data)
    length = str(len(stringData))
    conn.sendall(length.encode('utf-8').ljust(64))
    conn.send(stringData)
    conn.send(stime.encode('utf-8').ljust(64))
    '''
    print("width = ", frame.shape)
    time.sleep(1)
    #another type of code
    ## take initial image size from cam and resize img on client side while displaying 
    #ret, frame = capture.read()
    #print(frame)
    frame = imutils.resize(frame, width = 380)
    a = pickle.dumps(frame)
    message = struct.pack("Q", len(a)) + a
    conn.sendall(message)
    #print(a)
    print("image send from server")
    #return ret
    print("replace text file")
    ######### src and dst
    print("1")
    with open('parameter.txt', 'w+') as f:
      print("2")
      #server_socket.settimeout(1000)
      #print("current timeout=",server_socket.gettimeout())
      
      try:
        src = conn.recv(2048)
        src = pickle.loads(src)
        print("src recieved",src)
        print("3")
        dst = conn.recv(2048)
        dst = pickle.loads(dst)
        print("dst recieved",dst)
        ############# x_image , y_image
        x_image = conn.recv(2048)
        print("4")
        x_image = pickle.loads(x_image)
        print("x_I recieved",x_image)
        y_image = conn.recv(2048)
        y_image = pickle.loads(y_image)
        print("Y_I recieved",y_image)
        ######## email and list of veh
        email = conn.recv(2048)
        print("6")
        email = pickle.loads(email)
        print("email recieved",email)
        list_of_vehicles = conn.recv(2048)
        list_of_vehicles = pickle.loads(list_of_vehicles)
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
          if len(y_image) == 0:
              print ('y 0')
              #sys.exit(0)
      print("5")
      print("recv data",src,dst,list_of_vehicles)
      print("LOV ", list_of_vehicles)
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
      print("3")
      #server_socket.settimeout(10)
      #print("current timeout 2=",server_socket.gettimeout())

conn = False
video_num = 1
while True:
    try:
      #server_socket.settimeout(10)
      conn, address = server_socket.accept()
      print("conn", conn)
      if conn:
          print("timer time changed to 120")
          print("recieved connection from", address)
          task = conn.recv(1024).decode('utf-8')
          print(task)
          if not task:
              print("nahi")
              Text_file = 10
          elif task == "1":
              print("sending image")
              #receiveThread = threading.Thread(target=sendImages)
              #receiveThread.start()
              sendImages(conn)
              print("image send from server")
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

    #except socket.timeout:
    except Exception as e:
      print(e)
      time.sleep(10)
    '''
    Check = os.path.isfile('parameters.txt')
    print("check =", Check)
    print("running main code")
    if Check == True:
        with open('parameters.txt') as f:
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

    current_date = datetime.now()
    times_now = int(current_date.strftime("%H%M%S"))
    if  times_now <= 180000 and times_now >= 90000:
        #check file name
        path_file = 'save_videos/'+ str(video_num)+'.mp4'
        Check = os.path.isfile(path_file)
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
            email_session.sendmail('autonomoustrafficanalysis@gmail', 'pareespathak@gmail.com',my_message)
            ## prepare to send
            print(video_num)
            try:
              os.remove(file)
              video_num = video_num + 1
            except:
              print("video_num cannot be deletted ", video_num + 1)

            ## delete video file

    ########### sending sheets
    if times_now >= 220000 and times_now <= 240000:
      file_size = 0 
      length = os.listdir('sheets')
      #print(len(length))
      for i in range (0, len(length)):
          save_name = 'sheets/generated output' + str(i)+'.xls'
          if os.path.isfile(save_name) == True:
              file_size = file_size + (os.path.getsize(save_name)/1000000)
              #print(file_size)
              if file_size < 20:
                  sheet_attach = open(save_name, 'rb')
                  sheet_obj = MIMEBase('application','octet-stream')
                  sheet_obj.set_payload((sheet_attach).read())
                  encoders.encode_base64(sheet_obj)
                  sheet_obj.add_header('Content-Disposition',"attachment; filename= "+save_name)
                  msg.attach(sheet_obj)

              if file_size >= 20 or i == (len(length) -1):
                  print(len(length))
                  print("send")
                  my_message = msg.as_string()
                  email_session.sendmail('autonomoustrafficanalysis@gmail', emailid, my_message)
                  msg = MIMEMultipart()
      try:
        shutil.rmtree('sheets')
        os.mkdir('sheets')
      except:
        time.sleep(10)
      ### delete folder 
      ## recreate sheets folder for next day
      '''
 
    print("end of connection")