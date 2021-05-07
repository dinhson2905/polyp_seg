import os
import argparse
from datetime import datetime
import configparser
import logging
from logs import setup_logger

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from libs.efficientnetv2 import EffNetV2SCPD
from losses.region_loss import WIoUBCELoss
from utils.data_loader import get_loader, ValidationDataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter

def validation(model, data_path):
    model.eval()
    image_root = f'{data_path}/images/'
    gt_root = f'{data_path}/masks/'
    validation_loader = ValidationDataset(image_root, gt_root, 352)
    b = 0.0
    for i in range(validation_loader.size):
        image, gt, name = validation_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=True)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input_ = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input_, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        loss = (2 * intersection.sum() + smooth) / (input_.sum() + target.sum() + smooth)
        a = '{:4f}'.format(loss)
        a = float(a)
        b = b + a
    return b / validation_loader.size


def train(train_loader, model, optimizer, epochs, batch_size, train_size, clip, test_path):
    best_dice_score = 0.0
    for epoch in range(1, epochs):
        adjust_lr(optimizer, lr, epoch, 0.1, 200)
        for param in optimizer.param_groups:
            print(param['lr'])
            train_logger.info('learning rate: ', param['lr'])
        model.train()
        size_rates = [0.75, 1, 1.25]
        loss_record = AvgMeter()
        criterion = WIoUBCELoss()
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                trainsize = int(round(train_size * rate / 32) * 32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                #predict
                predict_maps = model(images)
                loss = criterion(predict_maps, gts)
                loss.backward()
                clip_gradient(optimizer, clip)
                optimizer.step()

                if rate == 1:
                    loss_record.update(loss.data, batch_size)
            if i%20 == 0 or i == total_step:
                print(f'{datetime.now()} Epoch [{epoch}/{epochs}], Step [{i}/{total_step}], Loss: {loss_record.show()}')
                train_logger.info(f'{datetime.now()} Epoch [{epoch}/{epochs}], Step [{i}/{total_step}], Loss: {loss_record.show()}')
        
        save_path = 'checkpoints/'
        os.makedirs(save_path, exist_ok=True)

        if (epoch+1) % 1 == 0:
            meandice = validation(model, test_path)
            print(f'meandice: {meandice}')
            train_logger.info(f'meandice: {meandice}')
            if meandice > best_dice_score:
                best_dice_score = meandice
                torch.save(model.state_dict(), save_path + 'effnetv2scpd_2704.pth')
                print('[Saving Snapshots:]', save_path + 'effnetv2scpd_2704', meandice)




if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    optimizer_type = config['Parameter']['optimizer']
    lr = float(config['Parameter']['lr'])
    batch_size = int(config['Parameter']['batch_size'])
    epochs = int(config['Parameter']['epochs'])
    train_size = int(config['Parameter']['train_size'])
    augumentation = config['Parameter'].getboolean('augumentation')
    clip = float(config['Parameter']['clip'])
    train_path = config['Paths']['train_path']
    test_path = config['Paths']['test_path']

    test_kvasir_path = f'{test_path}/Kvasir'
    train_logger = setup_logger('train_logger', 'logs/train_effnetv2_2704.log')

    model = EffNetV2SCPD()
    weights = torch.load('checkpoints/effnetv2scpd_')
    model.load_state_dict(weights)
    model = model.cuda()
    params = model.parameters()
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(params, lr)
    else:
        optimizer = torch.optim.SGD(params, lr, weight_decay=1e-4, momentum=0.9)

    image_root = '{}/images/'.format(train_path)
    gt_root = '{}/masks/'.format(train_path)
    
    train_loader = get_loader(image_root, gt_root, batchsize=batch_size, trainsize=train_size, augmentation=augumentation)
    total_step = len(train_loader)

    print('#'*20, 'Start Training', '#'*20) 
    train(train_loader, model, optimizer, epochs, batch_size, train_size, clip, test_kvasir_path) 