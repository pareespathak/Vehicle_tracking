# Vehicle_tracking 
download weights of yolov4 from darknet  
video from drive link or pc


Anaconda GPU Software Installation Guide with Python

Introduction:

This manual serves as a step-by-step guide to help you successfully install software on your computer. Whether you're a beginner or an experienced user, this guide will provide you with the necessary instructions and best practices to ensure a smooth installation process. Just so you know, the specific steps may vary depending on the software you are installing, but this guide will provide you with a general framework to follow.

For installing Anaconda with GPU support, specifically tailored for Python development. By following these instructions, you'll be able to set up a Python environment that leverages GPU capabilities, enabling accelerated computation for data science, machine learning, and other GPU-intensive tasks.

Table of Contents:
1. System Requirements
2. Pre-installation Checklist
3. Downloading Anaconda
4. Installing Anaconda
5. Configuring Anaconda for GPU Support
6. Testing the Installation
7. Installing other libraries manually and through the command prompt 
8. Downloading Yolo Files
9. Starting the program
10. Conclusion

1. System Requirements:
Before proceeding, ensure that your system meets the minimum requirements for running Anaconda with GPU support. Verify the compatibility of your GPU with CUDA, as well as the necessary drivers and operating system version. Refer to the documentation provided by Anaconda for specific system requirements.

2. Pre-installation Checklist:
To prepare for the installation, follow these pre-installation steps:

- Update GPU drivers: Make sure your GPU drivers are up to date. Visit the official website of your GPU manufacturer to obtain the latest drivers.
- Verify CUDA compatibility: Check if your GPU supports the required CUDA version. Refer to the CUDA documentation for compatibility details.
- Uninstall previous CUDA installations: If you have a previous version of CUDA installed, it is recommended to uninstall it before proceeding.
- Backup important data: Create a backup of any critical files or configurations to prevent data loss during the installation process.

3. Downloading Anaconda:
To download Anaconda, follow these steps:

- Step 1: Visit the Anaconda website (https://www.anaconda.com/products/individual) and navigate to the Individual Edition section. For windows installation refer (https://docs.anaconda.com/free/anaconda/install/windows/)
- Step 2: Select the appropriate installer for your operating system (Windows, macOS, or Linux) and click on the download button.
- Step 3: Wait for the download to complete. This may take some time depending on your internet connection.

4. Installing Anaconda:
Once the Anaconda installer is downloaded, follow these installation steps:

- Step 1: Run the installer executable file.
- Step 2: Follow the on-screen instructions to proceed with the installation.
- Step 3: Choose the installation directory and select whether to add Anaconda to your system's PATH environment variable (recommended).
- Step 4: Complete the installation process by clicking on the "Install" button.
- Step 5: Wait for the installation to finish. This may take a few minutes.
- - If you face any problem follow this guide (https://medium.com/@GalarnykMichael/install-python-on-windows-anaconda-c63c7c3d1444)

5. Build OpenCV with CUDA:

Follow the guide (https://youtu.be/YsmhKar8oOc)
To ensure the correct version installations download the corresponding versions mentioned in the links.
- Now open the Anaconda terminal and run the following codes.
    ```
    conda create -n ATMAS python=3.6
    conda activate ATMAS
    pip install PyQt5
    conda install -c anaconda numpy
    conda install -c anaconda scipy
    conda install -c conda-forge opencv=4.5.1
    conda install -c anaconda xlwt
    conda install -c conda-forge matplotlib
    ```
   
- Step 3: Activate the newly created environment whenever using the program:

7. Setting Up Yolov4 weights

To download YOLOv4 weights and other files for Python, follow the steps outlined below:
- Access the Darknet repository: YOLOv4 is implemented in Darknet, an open-source framework. Visit the Darknet repository on GitHub at (https://github.com/AlexeyAB/darknet).
- Clone or download the Darknet repository: On the Darknet repository page, click the "Code" button, then select the appropriate option to clone the repository using Git or download it as a ZIP file. Choose the option that suits your preference.
- Extract the downloaded ZIP file: If you downloaded the repository as a ZIP file, extract it to a directory of your choice using a file extraction tool.
- Download the YOLOv4 weights: Inside the Darknet repository, locate the "cfg" folder and navigate to it. Within the "cfg" folder, you'll find the pre-configured YOLOv4 model configurations. You can also download the pre-trained YOLOv4 weights from the official Darknet website at https://github.com/AlexeyAB/darknet/releases/tag/darknet_yolo_v3_optimal. 
- Download the file named "yolov4.weights" and save it to the same directory as the Darknet repository.
- Additional files and configurations include the YOLOv4 configuration file ("yolov4.cfg"), and class labels file ("coco.names"). These files are available within the Darknet repository's "cfg" folder. Download the necessary files based on your specific use case.

For reference, you can download the files from the source drive page of our repository.

8. Starting the code
