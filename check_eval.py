import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import configparser

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    test_path = config['Paths']['test_path']
    result_path = config['Paths']['result_path']

    