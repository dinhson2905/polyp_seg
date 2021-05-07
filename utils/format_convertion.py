import os
import shutil
from libtiff import TIFF
from scipy import misc
import random

def tif2png(_src_path, _dst_path):
    tif = TIFF.open(_src_path, mode='r')
    image = tif.read_image()
    misc.imsave(_dst_path, image)

def data_split(src_list):
    counter_list = random.sample(range(0, len(src_list)), 550)
    return counter_list
