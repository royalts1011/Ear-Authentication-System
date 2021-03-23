# Ear-Authentication-System
This repository contains all created components of our bachelor thesis.
The main components are:
- Ear Detector with the possibility to acquire a dataset.
- Ear authentication system.

# How to set up this project

## 0. Cloning, Python & Pip
- Clone the repo. We used Documents as the base folder.
- Make sure Python and Pip is installed. We developed this project using Python 3.7.7, the Raspberry Pi used Python 3.7.3

## 1. Configure _~/.profile_ in Linux
- Open terminal and execute `nano ~/.profile` to open and edit file
- Open the _profile_setup.txt_ and copy all lines into the _.profile_ file without deleting the already existing content
> **Note:** The line `export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3` refers to python3 being installed.
> In the next step we therefore use pip3 in the commands.

## 2. Virtual Environment
1. Install virtualenv and virtualenvwrapper for Python: https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/
	* `cd ~ && pip3 install virtualenv && pip3 install virtualenvwrapper`
	> **Note:** You could also install it via normal pip if the _.profile_ refers to _python_ instead _python3_
2. Create an environment with the python 3.7 version
	* `mkvirtualenv env_name -p 3.7`
3. To activate your created virtualenv execute these commands in that order:
	* `source ~/.profile`
	* `workon env_name`
	> **Note:** Every time you want to use the environment you need to execute this step 2.3 
4. To exit an environment use `deactivate`

## 3. Configure the Virtual Environment for the project
- Navigate to the repository. For the next step **check that you are in the virtual environment!**
- You will find a _requirements.txt_ file in the repository. To easily set up all required python packages execute following line:
	* `pip install -r requirements.txt`


## Working with the general project should now be possible. Detailed Information to the folders can be found in their location respectively.
