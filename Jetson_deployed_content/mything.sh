#!/bin/bash
#!/usr/bin/python3
sleep 10

xterm -e python3 /home/parees/Downloads/Vehicle_tracking-main/Jetson/device/device_server.py &
xterm -e python3 /home/parees/Downloads/Vehicle_tracking-main/Jetson/device/device_mail.py &
xterm -e python3 /home/parees/Downloads/Vehicle_tracking-main/Jetson/device/device_code.py &

exit 0
