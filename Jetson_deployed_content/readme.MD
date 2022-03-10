############# From start 
Build opencv
### Increase swap memory to 8 gb for build



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



create a service file to boot at startup
path of lib should be specific (install using: sudo pip3 install name0

