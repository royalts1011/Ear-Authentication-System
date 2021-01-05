import cv2
import os
from os.path import join, dirname, exists
import time

#########################################################################
# SET PARAMETERS
#########################################################################

# set amount of pictures and pictures per step setting
PICTURES  = 80
STEP = 20

DATASET_DIR = '../dataset'
GREEN = (0,255,0)


# additional space around the ear to be captured
# 0.1 is tightly around, 0.2 more generous 
SCALING_H = 0.05
SCALING_W = 0.2

print(  "\n [INFO]\n",
        "------------------------------\n",
        "------------------------------\n",
        "In this process {amount:} pictures will be taken. Throughout the process, the ear should be in different positions.\n".format(amount=PICTURES),
        "Per position the detector will take {step:} continuous shots and will then wait for user interaction to initiate the next position.\n".format(step=STEP),
        "------------------------------\n",
        "------------------------------\n",
        "The detected ear bounding box will have a larger height by {height:.2%} and a larger width by {width:.2%}.\n".format(height=2*SCALING_H, width=2*SCALING_W),
        "------------------------------\n",
        "------------------------------\n",
        "The images will be saved in a folder in the directory '{dir:}'.\n".format(dir=DATASET_DIR),
        "------------------------------\n",
        "------------------------------"
        )
#########################################################################



#########################################################################
# [INITIAL PROCEDURE OF HEAD TURNING] 
#         Initializing ear capture. Turn your head left. Your right ear should then be facing the camera.
#         Look into the camera and slowly turn your head 45 degrees to the left
#         Now look up, keeping the right ear towards the camera.
#         Now look down, keeping the right ear towards the camera.
#########################################################################
# assert PICTURES/10 <= (len(user_instructions))


"""    
    These functions can be used to set the camera to a certain resolution.
    The object is expected to be a cv2.VideoCapture() type.
"""
def make_720(object):
    object.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    object.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
def make_540(object):
    object.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    object.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
def make_480(object):
    object.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    object.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
def make_240(object):
    object.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    object.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


""" This function can be used to scale a frame by a certain percentage.

    Args:
        frame: The frame/image
        percent: Percentual Resizing
    Returns:
        The resized frame
"""
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)


""" Main function for the detection and saving the ear images,
    If is_authentification is true, the function is used for creating new embeddings which are based on just one image.
    The created files are deleted by the script that is calling this function.

    Args:
        amount_pic: total amount of images
        pic_per_stage: amount of images per position
        is_authentification: boolean if function is used for authentification
"""
def capture_ear_images(amount_pic=PICTURES, pic_per_stage=STEP, is_authentification=False):

    cap = cv2.VideoCapture(0)
    time.sleep(2.0)
    # open window dimensions
    make_720(cap)

    ear_detector = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')

    # For each person, enter a new identification name
    print("The format of names should be consistent. (e.g. 'firstname_lastname' only using first three letters of the last name. --> 'max_mus'")
    ear_name = input('\n Enter name end press <return> :\t')

    if not exists(DATASET_DIR):
        os.mkdir(DATASET_DIR)

    usr_dir = (join(DATASET_DIR, ear_name), join(DATASET_DIR, (ear_name + '-auth')))[is_authentification]
    if not exists(usr_dir):
        os.mkdir(usr_dir)
        
    # Initialize individual sampling ear count
    count = 0

    while True:
        # receive image
        ret, frame = cap.read()
        # flip video frame horizontally to show it "mirror-like"
        frame = cv2.flip(frame, 1)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = ear_detector.detectMultiScale(grey, 1.1, 5)

        for (x,y,w,h) in rects:
            # bounding box will be bigger by increasing the scaling
            left = x - int(w * SCALING_W)
            top = y - int(h * SCALING_H)
            right = x + int(w * (1+SCALING_W))
            bottom = y + int(h * (1+SCALING_H))
            cv2.rectangle(frame, (left, top), (right, bottom), color=GREEN, thickness=1)   
            count += 1

            cv2.imshow('Frame', frame)

            frame = frame[top+1:bottom, left+1:right] # +1 eliminates rectangle artifacts
            # Re-flip image to original
            frame = cv2.flip(frame, 1)
            # Save the captured image into the datasets folder
            cv2.imwrite(join(usr_dir, (ear_name + "{0:0=3d}".format(count) + ".png")), frame)

            # display after defined set of steps 
            if (count%pic_per_stage) == 0 and count != amount_pic:
                print("\n [INFO] Next step commencing... \n")
                print(count)
                input("Reposition your head and press <return> to continue.")


        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= amount_pic: # Stop loop when the amount of pictures is collected
            print(count)
            break


    # Close all windows and end camera
    print("\n [INFO] Exiting Program.")
    cap.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    capture_ear_images()    
