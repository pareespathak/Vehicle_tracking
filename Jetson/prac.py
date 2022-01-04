from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
#from email.mime.multipart import MIMEBase 
import smtplib
from email.message import EmailMessage
#from email import Encoders
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

'''
smtp = smtplib.SMTP('smtp.gmail.com', 587)
smtp.ehlo()
smtp.starttls()
smtp.login('autonomoustrafficanalysis@gmail.com', 'Auto@124')

msg = EmailMessage()
msg['Subject'] = 'Auto'
msg['From'] = 'autonomoustrafficanalysis@gmail.com'
msg['To'] = 'pareespathak@gmail.com'

with open ('Outside Paint.avi', 'rb') as f:
    file_name = f.read()
    #msg.add_attachment(file_name, maintype = 'image', subtype = 'jpeg')
    #video/avi
    msg.add_attachment(file_name, maintype = 'video', subtype = 'x-msvideo')


smtp.send_message(msg)
'''


email_session = smtplib.SMTP('smtp.gmail.com',587)
email_session.starttls()
email_session.login('autonomoustrafficanalysis@gmail.com', 'Auto@124')

msg = MIMEMultipart()
msg['Subject'] = 'Hello'
msg['From'] = 'autonomoustrafficanalysis@gmail.com'
msg['To'] = 'pareespathak@gmail.com'

##############  Size less than 20 mb
file = "interaction 1.mp4"
attachment = open(file,'rb')
obj = MIMEBase('application','octet-stream')
obj.set_payload((attachment).read())
encoders.encode_base64(obj)
obj.add_header('Content-Disposition',"attachment; filename= "+file)

msg.attach(obj)
my_message = msg.as_string()

email_session.sendmail('autonomoustrafficanalysis@gmail', 'pareespathak@gmail.com',my_message)


'''
path_file = r'C:\aa\vehicle_tracking_college\tracking\Vehicle_Tracking_Program\GUI\socket\Jetson\reference.py'
print(path_file)
Check = os.path.isfile(path_file)
print("check =", Check)
if Check == True:
    print("file exists")
        #send text of file found
else:
    print("request file")
    #send text file not found, crea
'''
