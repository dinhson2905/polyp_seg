import os, argparse
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from libs.hardmseg import HarDCPD, HarDMSEG
from libs.pranet import PraNet
from libs.respd import ResNetPD, ResNetCPD
from utils.data_loader import ValidationDataset
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

test_size = int(config['Parameter']['test_size'])
test_path = config['Paths']['test_path']
model_name = config['Parameter']['model']
if model_name == 'pranet':
    model = PraNet()
    pth_path = config['Paths']['pranet_pth_path']
    result_path = config['Paths']['pranet_result_path']
elif model_name == 'hardmseg':
    model = HarDMSEG()
    pth_path = config['Paths']['hardmseg_pth_path']
    result_path = config['Paths']['hardmseg_result_path']
elif model_name == 'hardcpd':
    model = HarDCPD()
    pth_path = config['Paths']['hardcpd_pth_path']
    result_path = config['Paths']['hardcpd_result_path']
elif model_name == 'resnetpd':
    model = ResNetPD()
    pth_path = config['Paths']['resnetpd_pth_path']
    result_path = config['Paths']['resnetpd_pth_path']    
elif model_name == 'resnetcpd':
    model = ResNetCPD()
    pth_path = config['Paths']['resnetcpd_pth_path']
    result_path = config['Paths']['resnetcpd_pth_path']


model.load_state_dict(torch.load(pth_path))
model.cuda()
model.eval()


for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    data_path = f'{test_path}/{_data_name}'
    save_path = f'{result_path}/{_data_name}/'
    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = ValidationDataset(image_root, gt_root, test_size)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        if model_name == 'pranet':
            _, _, _, res = model(image)
        elif model_name == 'hardmseg' or model_name == 'resnetpd':
            res = model(image)
        elif model_name == 'hardcpd' or model_name == 'resnetcpd':
            _, res = model(image)
        
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        imageio.imwrite(save_path + name, res)