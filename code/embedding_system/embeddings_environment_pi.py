#other imports
import sys
sys.path.append('..')
import numpy as np
from PIL import Image
import glob
import os
import shutil
from time import sleep

# PyTorch imports
import torch
import torch.nn.functional as F
from torch import cuda
from torch.autograd import Variable
from torchvision.models.mobilenet import mobilenet_v2

# own script imports
from training.helpers import get_device
from training.helpers import cuda_conv
import training.ds_transformations as td
import training.metrics as M
import ear_detector.acquire_ear_dataset as a

# Pin imports
import board
import digitalio
import adafruit_character_lcd.character_lcd as character_lcd
from gpiozero import LED

# instantiate lcd and specify pins
lcd_rs = digitalio.DigitalInOut(board.D26)
lcd_en = digitalio.DigitalInOut(board.D19)
lcd_d4 = digitalio.DigitalInOut(board.D13)
lcd_d5 = digitalio.DigitalInOut(board.D6)
lcd_d6 = digitalio.DigitalInOut(board.D5)
lcd_d7 = digitalio.DigitalInOut(board.D11)
lcd_columns = 16
lcd_rows = 2
lcd = character_lcd.Character_LCD_Mono(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7, lcd_columns, lcd_rows)
# lcd = Adafruit_CharLCD(rs=26, en=19, d4=13, d5=6, d6=5, d7=11, cols=16, lines=2)

# initiate LEDs
led_yellow = LED(4)
led_green = LED(17)
led_red = LED(27)


class Config():
    """
    Configuration Class in which all necessary parameters that will be used in the further process are defined.
    """
    DEVICE = get_device()
    DATASET_DIR = '../../ear_dataset/'
    AUTH_DATASET_DIR = '../../auth_dataset/unknown-auth/'
    MODEL_DIR = '../../models/ve_g_margin_2,0.pt'
    is_small_resize = False
    DATABASE_FOLDER = '../../embeddings/'
    THRESHOLD_VAL = 0.9
    THRESHOLD = 2.0
    a = 0.1
    AUTHORIZED = ["falco_len.npy","konrad_von.npy"]


# Load the model that will be used to during the authentication process.
model = torch.load(Config.MODEL_DIR, map_location=torch.device(Config.DEVICE))
# Specify a set of transformations to be applied to all images during the authentication process.
transformation = td.get_transform('valid_and_test', Config.is_small_resize)


def pipeline(input_, preprocess):
    """
    This method performs a series of image processing procedures. It also checks whether one of the tensor in the
    following can be processed on the graphics card.
    1. convert the input to gray image
    2. perform preprocessing (in this case defined in the transformations
    3. sizes adjustment
    4. rearrange the tensor
    """
    input_ = input_.convert("L")
    input_ = preprocess(input_)
    input_ = input_.reshape(-1, td.get_resize(Config.is_small_resize)[0], td.get_resize(Config.is_small_resize)[1], 1)
    input_ = input_.permute(3, 0, 1, 2)

    if cuda.is_available():
        return input_.type('torch.cuda.FloatTensor')
    else:
        return input_.type('torch.FloatTensor')


# Verification
try:
    """
    At this point, 4 images are initially captured. Then the last image taken is used for authentication.
    This image is then also pre-processed and processed by the network. 
    As a result you receive 1 Embedding. This embedding is now compared with the entire embeddings database. 
    For each person, the smallest distance is stored.

    At this point, a better approach would be not to store the least distance of each person, 
    but to calculate the average of the distances to all embeddings of a person.
    This would make the system more robust against outliers.
    """
    lcd.blink = False
    
    # LCD output
    lcd.clear()
    lcd.message = 'Ready to take\nyour ear images'

    # Bilder aufnehmen
    led_yellow.blink(on_time=0.5,off_time=0.25)

    a.capture_ear_images(amount_pic=4, pic_per_stage=4, is_authentification=True)

    led_yellow.off()
    
    # LCD output
    lcd.clear()
    lcd.message = 'Verification\nin progress...'


    result_value = []
    result_label = []

    img = Image.open(Config.AUTH_DATASET_DIR + 'unknown004.png')
    new_embedding = model(Variable(pipeline(img,transformation))).cpu()

    for label in os.listdir(Config.DATABASE_FOLDER):
        loaded_embedding = np.load(Config.DATABASE_FOLDER+label, allow_pickle=True)
        tmp = []    
        for embedding in loaded_embedding:
            dis = F.pairwise_distance(embedding,new_embedding)
            tmp.append(dis.item())
        result_value.append((min(tmp)))
        result_label.append(label)


    # Listing of the 10 closest distances and the associated people.
    result_value, result_label = zip(*sorted(zip(result_value, result_label)))
    result_value = result_value[:10]
    result_label = result_label[:10]

    for idx, val in enumerate(result_label):
        print(str(idx+1) + ' : ' + ' ' + val + ' : ' + ' ' + str(result_value[idx]))



    # At this point, we will review 3 security features that we have chosen.
    if (result_value[0] + Config.a) < result_value[1] and result_value[0] <= Config.THRESHOLD_VAL and result_label[0] in Config.AUTHORIZED:
        lcd.clear()
        entry_string = 'Hi ' + result_label[0]
        lcd.message = 'Access granted\n'+ entry_string

        print("Access granted! Welcome "  + result_label[0] + "!")

        led_green.on()
        sleep(10)
        led_green.off()

    if result_value[0] > Config.THRESHOLD_VAL and result_value[0] < Config.THRESHOLD:
        lcd.clear()
        lcd.message = 'Not found -\nNo entry.'
        
        print('Cant find authorized Person in Database. Pls try again')

        led_red.on()
        sleep(10)
        led_red.off() 

    if (result_value[0] + Config.a) >= result_value[1]:
        lcd.clear()
        lcd.message = 'Access denied -\nNo entry.'
                
        print('Verification not clear enough. Access denied. Please try again.')
        led_red.on()
        sleep(10)
        led_red.off() 



finally:
    lcd.clear()

    # Delete the captured images, as they are no longer needed.
    if os.path.exists('../../auth_dataset/'):
        shutil.rmtree('../../auth_dataset/')

