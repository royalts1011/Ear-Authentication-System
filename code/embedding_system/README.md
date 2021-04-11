## _create_embedding.ipynb_
This Notebook can be used to transform your dataset of images to Embeddings. 
- Make sure your dataset has the following strcuture: **dataset/subject/img001.png**.
- Also make sure that you have created all required folders and set the pathes correctly.
- You will receive **1 Embedding file for each subject** which contains the vectors/embeddings for every Image inside the subjects folder



## _create_embedding_newperson.ipynb_
Quite simillar to the Notebooke above this Notebooks does pretty the same. The only difference is, that this one is used for a single subject, not for a whole dataset of subjects.
The reason behind this notebook is that after our system is trained it can also be used for unknown subject. The only thing you need is do create the Embedding File for a new Subject as a form of registration.
What this notebook does is **creating 1 Embedding file for 1 Subject** while checking if this subject allready exists.



## _embedding_testenvironment.ipynb_
You can use this Notebook to test out and build up the ear_recognition System. Unlike the following python scripts, for this notebook it is required to put at least on picture in the stored folder for **AUTH_DATASET_DIR**.

As a result you receive 1 Embedding. This embedding is now compared with the entire embeddings database. For each person, the smallest distance is stored.
At this point, a better approach would be not to store the least distance of each person, but to calculate the average of the distances to all embeddings of a person.
This would make the system more robust against outliers.

Finally this notebook gives you a list of the 10 closest distances and the associated people.
![Algorithm_Part_1](https://github.com/royalts1011/Ear-Authentication-System/blob/606c9ed8ed35b0197c8f2af0b9e882a01e92b7d6/code/embedding_system/Algorithm_part1.png)


## _embedding_environment.py_
What is still missing from the notebook is now included in this script.
This script can be executed on any computer that:
- has a camera
- has Python version 3.7.7 installed
- has installed all the packages we specified
In contrast to the notebook described before, this script takes the image to be processed on its own. The further processing steps remain the same for the time being.
However, the previously described list of the 10 closest distances, which belong to 10 different persons, is now used to make a final decision whether the authentication was successful or not.
![Algorithm_Part_2](https://github.com/royalts1011/Ear-Authentication-System/blob/606c9ed8ed35b0197c8f2af0b9e882a01e92b7d6/code/embedding_system/Algorithm_part2.png)


## _embedding_environment_pi.py_
Both scripts is literally the same. The pi script has only a few extension to controll LED, LCD, etc. of our builded hardware setup.
The packages surrounding RPi equipment such as the 16x2 LCD change quite often. Don't be alarmed when the code for Pin outputs does not work anymore. It is thought to be a sort of mockup for possible tiny and simple user feedback. The specified GPIO pins correspond our witing which we depicted in the following aimge:


