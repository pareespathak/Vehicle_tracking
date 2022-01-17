from email import encoders
import email
import os
import socket
import cv2
import numpy
import base64
import glob
import sys
from datetime import datetime
import pickle
from xlwt import Workbook
import csv
import numpy as np
import pytz
import smtplib
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase

'''email parameters 
'''
email_session = smtplib.SMTP('smtp.gmail.com',587)
email_session.starttls()
email_session.login('autonomoustrafficanalysis@gmail.com', 'Auto@124')
msg = MIMEMultipart()
msg['Subject'] = 'Hello'
msg['From'] = 'autonomoustrafficanalysis@gmail.com'
msg['To'] = 'pareespathak@gmail.com'
emailid = 'pareespathak@gmail.com'


file_size = 0 
length = os.listdir('sheets')
print(len(length))
for i in range (0, len(length)):
    save_name = 'sheets/generated output' + str(i)+'.xls'
    if os.path.isfile(save_name) == True:
        file_size = file_size + (os.path.getsize(save_name)/1000000)
        print(file_size)
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

#save_name = 'sheets/generated output' + str(i)+'.xls'
#file_size = os.path.getsize('sheets/data1.xls')
