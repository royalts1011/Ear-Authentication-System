# Ear-Authentication-System
This repository contains all created components of our bachelor thesis.
The main components are:
- Ear Detector with the possibility to acquire a dataset. (Works on PC and Raspberry Pi)
- Ear authentication system
	- Training (PC ... with good enough specs)
	- Verification via embeddings (PC & RPi)

# How to set up this project on Linux PC or Raspberry Pi 4B

## 0. Cloning, Python & Pip
- Clone the repo. We used Documents as the base folder.
- Make sure Python and Pip is installed. We developed this project using Python 3.7.7, the Raspberry Pi used Python 3.7.3
- The camera module must be activated in RPi Settings for several scripts in this project

## 1. Configure _~/.profile_ in Linux and RPi
- In the terminal execute `nano ~/.profile` to open and edit file
- Open the _profile_setup.txt_ and copy all lines into the _.profile_ file without deleting the already existing content
> **Note:** The line `export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3` refers to python3 being installed.
> In the next step we therefore use pip3 in the commands.

## 2. Virtual Environment
1. Install virtualenv and virtualenvwrapper for Python: https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/
	* `cd ~ && pip3 install virtualenv && pip3 install virtualenvwrapper`
	> **Note:** You could also install it via normal pip if the _.profile_ refers to _python_ instead _python3_ 
2. Create an environment with the corresponding python version (we use 3.7). Make sure to reload the _.profile_ first.
	* `source ~/.profile`
	* `mkvirtualenv env_name -p python3`
3. To activate your created virtualenv (e.g. upon starting a new terminal) execute these commands in that order:
	* `source ~/.profile`
	* `workon env_name`
	> **Note:** Every time you want to use the environment you need to execute this step 2.3 
4. To exit an environment use `deactivate`

## 3. Configure the Virtual Environment for the project. <br /> Choose part A _OR_ part B

### Part A: You are on a Linux PC and NOT the Raspberry Pi
- Navigate to the repository. For the next step **check that you are IN the virtual environment!**
- You will find a _requirements.txt_ file in the repository. To easily set up all required python packages execute following line:
	* `pip install -r requirements.txt`

### Part B: You are on the Raspberry Pi 4B
> **Note:** This is specifically RPi **4B**, as the torch and torchvision PiWheel was compiled on and build for RPi 4B
1. For image manipulation and in first line for the Cascade Classifier we need OpenCV on the Raspberry Pi. We need dependencies, triggers and a functioning build. Follow **Step #2 and Step #4a** of this tutorial. Note that the dependencies (Step #2) are done **OUTSIDE** the virtual environment. The opencv installation (Step #4a: `pip install opencv-contrib-python==4.1.0.25`) is done **INSIDE** the virtual environment: https://www.pyimagesearch.com/2019/09/16/install-opencv-4-on-raspberry-pi-4-and-raspbian-buster/
	> We used the full install of OpenCV 4 (Step #4b) and seemed to have achieved a lower latency in the real time ear detection than with the quick install. Quick isntall does work fine though.

2. We will need certain torch and torchvision dependencies and pre-build wheel files. If you are on another Raspberry Pi version this is the step where you will need to find fitting wheels for your soon to be torch-vision lightning speed car.
* Install dependencies for PyTorch **OUTSIDE** the virtual environment.
	* `sudo apt install libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools`
	> Fetched from https://medium.com/secure-and-private-ai-writing-challenge/a-step-by-step-guide-to-installing-pytorch-in-raspberry-pi-a1491bb80531
* Clone repo or download the two wheel-files from the following repo. Install both wheels using pip **INSIDE** the virtual environment. https://github.com/sungjuGit/PyTorch-and-Vision-for-Raspberry-Pi-4B
	* `pip install xxx.whl`

3. Now you can proceed and install the _requirements_pi.txt_ fro this repo. It contains a different packages like _picamera_ or RPi-specifics.
	* `pip install -r requirements_pi.txt`
	> **Note:** If you choose to update certain packages (newer may exist) be careful with torch, torchvision and exclude opencv from updates - or only update to a version working on the RPi (not all do). https://piwheels.org/project/opencv-contrib-python/#install
 

## Working with the project should now be possible. Detailed Information to the folders can be found in their location respectively.

> Full install of OpenCV 4: https://www.pyimagesearch.com/2019/09/16/install-opencv-4-on-raspberry-pi-4-and-raspbian-buster/
> This article from Step 2 contains further links to torch & torchvision versions: https://medium.com/secure-and-private-ai-writing-challenge/a-step-by-step-guide-to-installing-pytorch-in-raspberry-pi-a1491bb80531
