# Ear-Authentication-System
This repository contains all created components of our bachelor thesis.
The main components are:
- Ear Detector with the possibility to acquire a dataset.
- Ear authentication system.

# How to set up this project.

# 1. Configure .profile in Linux
- open terminal and use "nano .profile" top open file
- open the "profile_setup.txt and copy all lines into the .profile file without deleting the already existing content:
- when ever you want to use this project open terminal first and type: "source .profile"

# 2. Copy DLBio to the folder you used in your .profile file
- You can find the DLBio folder in this repository as well

# 3. Configure virtual environment for project.
- Make Sure to have python 3.7.7 installed. Maybe another version works as well but we developed this project using python 3.7.7
- Install virtualenv for python: https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/
- to activate your created virtualenv you can also open a terminal type "source .profile" and afterwords "workon virtualenvName" 
- To easily set up all required python packages we have used you can use "pip install -r requirements.txt" . You will find 
the requirements.txt in the repository.


Working with the project should now be possible. In some places further changes are necessary, for example to use everything on the Rasberry Pi. Detailed instructions can be found in the folders.