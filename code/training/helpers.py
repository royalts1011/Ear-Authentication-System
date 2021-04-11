"""

  Script to outsource (helping) functions
  
"""
import os
import torch
from torch import cuda
import numpy as np


def print_list(list_):
    '''
        Method for displaying a list of files with an enumerated index
    '''
    fmt = '{:<8}{:<20}'
    print(fmt.format('Index', 'Name'))
    for i, name in enumerate(list_):
        print(fmt.format(i, name))


def rm_DSStore(list_):
    '''
        Removes .DS_Store string from a list
        (Apple Mac precautions e.g. when loading all files from a folder)
    '''
    return list(filter(('.DS_Store').__ne__, list_))


def choose_folder(dataset_path, name='unknown'):
    '''
        Takes a path input and shows all folders
        Let's user choose a folder by index
    '''
    folders = os.listdir(dataset_path)
    # remove all .DS_Store entries
    folders = rm_DSStore(folders)
    folder_name = name
    # if statement starts, when no name was given
    if folder_name == 'unknown':
        print_list(folders)
        # Handle the user's input for user name
        while True:
            idx = input('Choose your folder name by index: ')
            try:
                idx = int(idx)
                assert idx < len(folders) and idx >= 0
                break
            except (ValueError, AssertionError):
                print('The input was a string or not in the index range.')
        folder_name = folders[idx]
    
    assert folder_name in folders, 'The name was not found in the given folder: ' + dataset_path
    return folder_name


def cuda_conv(obj):
    '''
        Converts to cuda if possible
    '''
    if cuda.is_available():
        return obj.cuda()
    else:
        return obj


def get_device():
    '''
        Return device (cuda or cpu)
    '''
    device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
    return device


def get_num_params(model, count_only_trainable=True):
    '''
        Return number of trainable parameters
    '''
    def select(p):
        return p.requires_grad or not count_only_trainable

    model_p = [p for p in model.parameters() if select(p)]
    num_params = sum([np.prod(p.size()) for p in model_p])
    return num_params