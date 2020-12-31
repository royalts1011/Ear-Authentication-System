# Ear-Authentication-System
This repository contains all created components of our bachelor thesis.
The main components are:
- Ear Detector with the possibility to acquire a dataset.
- Ear authentication system.

# How to set up this project.

# 1. Configure .profile in Linux
- open terminal and use "nano .profile" top open file
- copy the following lines into the .profile file without deleting the already existing content:
	export WORKON_HOME=$HOME/.virtualenvs
	export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
	source ~/.local/bin/virtualenvwrapper.sh
	export PYTHONPATH=~/Documents/Bachelorarbeit/:$PYTHONPATH
- when ever you want to use this project open terminal first and tipe: "source .profile"