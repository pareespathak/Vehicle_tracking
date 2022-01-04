import os
import socket
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

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

Socket_IP = 'localhost'
Socket_Port = 8080
## bind 
server_socket.bind((Socket_IP, Socket_Port))
### listen 
server_socket.listen(1)
print("listning to client ")

def sendImages():
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = capture.read()
    #resize_frame = cv2.resize(frame, dsize=(480, 315), interpolation=cv2.INTER_AREA)
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
while True:
    Text_file = 10
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
        
        elif task == "2":
            ######### previous data 
            print("checking text file")
            path_file = r'C:\aa\vehicle_tracking_college\tracking\Vehicle_Tracking_Program\GUI\socket\parameters.txt'
            print(path_file)
            Check = os.path.isfile(path_file)
            print("check =", Check)
            if Check == True:
                print("file exists")
                conn.send("a".encode('utf-8'))
                #send text of file found
                #Text_file = 10
            else:
                print("request file")
                #send text file not found, crete filr 
                conn.send("b".encode('utf-8'))
                Text_file = 12

        else:
            Text_file = 10 
    
        '''
        if Text_file == 10:
            print("extract data from current file")
            task = conn.recv(1024).decode('utf-8')
            print(task)
        '''
        #else:
        print("replace text file")
        src = conn.recv(2048)
        src = pickle.loads(src)
        print("src",src)
        dst = conn.recv(2048)
        dst = pickle.loads(dst)

        email = conn.recv(2048)
        email = pickle.loads(email)

        list_of_vehicles = conn.recv(2048)
        list_of_vehicles = pickle.loads(list_of_vehicles)
        print("email", email, "dst", dst)
        file = open("parameters.txt", "w+")
        file.write(str(src)+ "\n")
        file.write(str(dst) + "\n")
        file.write(str(list_of_vehicles) + "\n")
        file.write(email + "\n") 
        file.close()
        '''
        # Workbook is created

        wb = Workbook()
        # add_sheet is used to create sheet.
        sheet1 = wb.add_sheet("sheet 1", cell_overwrite_ok=True)
        sheet1.write(0, 1, 'src')
        sheet1.write(0, 2, 'dst')
        sheet1.write(0, 3, 'list of veh')
        sheet1.write(0, 4, 'email')
        sheet1.write(0, 5, 'address')
        sheet1.write(1, 1, src)
        sheet1.write(1, 2, dst)
        sheet1.write(1, 3, list_of_vehicles)
        sheet1.write(1, 4, address)
        wb.save('Parameters.xls')
        '''




