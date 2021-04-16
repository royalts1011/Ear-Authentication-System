
#other imports
import sys
sys.path.append('..')
import numpy as np
from PIL import Image
import glob
import os
import shutil

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


"""
At this point, 4 images are initially captured. Then the last image taken is used for authentication.
 This image is then also pre-processed and processed by the network. 
As a result you receive 1 Embedding. This embedding is now compared with the entire embeddings database. 
For each person, the smallest distance is stored.

At this point, a better approach would be not to store the least distance of each person, 
but to calculate the average of the distances to all embeddings of a person.
This would make the system more robust against outliers.
"""
# capture ear images
a.capture_ear_images(amount_pic=4, pic_per_stage=4, is_authentification=True)
result_value = []
result_label = []

img = Image.open(Config.AUTH_DATASET_DIR + 'unknown004.png')
new_embedding = model(Variable(pipeline(img,transformation))).cpu()

for label in os.listdir(Config.DATABASE_FOLDER):
    if label.endswith(".npy"):
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
if result_value[0] > Config.THRESHOLD_VAL and result_value[0] < Config.THRESHOLD:
    print('Cant find authorized Person in Database. Pls try again')

if (result_value[0] + Config.a) >= result_value[1]:
    print('Verification not clear enough. Access denied. Please try again.')

if (result_value[0] + Config.a) < result_value[1] and result_value[0] <= Config.THRESHOLD_VAL:
    print("Access granted! Welcome "  + result_label[0] + "!")


# Delete the captured images, as they are no longer needed.
shutil.rmtree('../../auth_dataset/')

