import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
# from scipy import misc
import imageio
from libs.hardmseg.hardmseg import HarDCPD
from utils.data_loader import ValidationDataset
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

test_size = int(config['Parameter']['test_size'])
cpd_pth_path = config['Paths']['pth_path']
test_path = config['Paths']['test_path']
cpd_result_path = config['Paths']['cpd_result_path']

for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    data_path = f'{test_path}/{_data_name}'
    save_path = f'{cpd_result_path}/{_data_name}/'

    model = HarDCPD()
    model.load_state_dict(torch.load(cpd_pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = ValidationDataset(image_root, gt_root, test_size)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # output model have 2 values: attention_map, detection_map
        _, res = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        imageio.imwrite(save_path + name, res)
    
    