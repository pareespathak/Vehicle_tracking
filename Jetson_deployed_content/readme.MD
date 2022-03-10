############# From start 
##### Add fan start commadn at rc.local

> $sudo nano /etc/rc.local  

#and paste this inside  

> #!/bin/bash  
> sleep 10   
> #sudo /usr/bin/jetson_clocks  
> sudo sh -c ‘echo 255 > /sys/devices/pwm-fan/target_pwm’  
> exit 0  

Uncoment jetson_clocks if you need it at startup



### Build opencv
#### Increase swap memory to 8 gb for build
Follow jetson hacks video to build 


How to make something run on boot with systemd systems:
Create a file named “/usr/local/bin/mything.sh” with the commands you need to run on start:
> $ sudo nano /usr/local/bin/mything.sh  
> ...  
> $ sudo chmod +x /usr/local/bin/mything.sh  

Create a file called “/etc/systemd/system/mything.service”  
In this file, put this text:    

[Unit]  
Description=Example systemd service.  
After=network.target  

[Service]  
#Type=simple  
Type=oneshot  
ExecStart=/bin/bash /usr/local/bin/mything.sh  

[Install]  
WantedBy=multi-user.target  

### add absolute path and check to run as root

path of lib should be specific (install using: sudo pip3 install name0

