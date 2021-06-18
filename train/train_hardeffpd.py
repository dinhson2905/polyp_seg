import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils.data_loader import get_loader, ValidationDataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np

import logging
from logs import setup_logger
from libs.efficientnetv2 import HarDEffPD
from losses.region_loss import WIoUBCELoss
import configparser

def validation(model, data_path, data2_path):
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
    
    image2_root = f'{data2_path}/images/'
    gt2_root = f'{data2_path}/masks/'
    validation2_loader = ValidationDataset(image2_root, gt2_root, 352)
    b2 = 0.0
    for i in range(validation2_loader.size):
        image, gt, name = validation2_loader.load_data()
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
        b2 = b2 + a
    
    return b / validation_loader.size, b2 / validation2_loader.size

def train(train_loader, model, optimizer, epochs, batch_size, train_size, clip, test_path, test2_path):
    best_dice_score = 0
    best_clinic_score = 0
    for epoch in range(1, epochs):
        adjust_lr(optimizer, lr, epoch, 0.1, 50)
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
                trainsize = int(round(train_size * rate / 32) *32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # predict    
                res_map = model(images)
                loss = criterion(res_map, gts)
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
            meandice, clininc_score = validation(model, test_path, test2_path)
            print(f'meandice: {meandice}, {clininc_score}')
            train_logger.info(f'meandice: {meandice}')
            train_logger.info(f'clinic_score: {clininc_score}')
            if meandice > best_dice_score:
                best_dice_score = meandice
                torch.save(model.state_dict(), save_path + 'hardeffpd_kvasir.pth')
                print('[Saving Snapshots:]', save_path + 'hardeffpd_kvasir.pth', meandice)
            if clininc_score > best_clinic_score:
                best_clinic_score = clininc_score
                torch.save(model.state_dict(), save_path + 'hardeffpd_clinic.pth')
                print('[Saving Snapshots:]', save_path + 'hardeffpd_clinic.pth', clininc_score)
        if epoch in [50, 60, 70, 80, 90]:
            file_ = 'hardeffpd_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), save_path + file_)
            print('[Saving Snapshots:]', save_path + file_, meandice, clininc_score)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

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
    test_clinic_path = f'{test_path}/CVC-ClinicDB'
    log_file = 'logs/train_hardcpd_' + datetime.now().strftime('%Y%m%d%H') + '.log'
    train_logger = setup_logger('train_logger', log_file)
    model = HarDEffPD().cuda()
    print('Params: ', get_n_params(model))
    train_logger.info(f'Params: {get_n_params(model)}')
    train_logger.info(f'optimizer: {optimizer_type}, lr: {lr}, batch_size: {batch_size}, image_size: {train_size}')
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
    train(train_loader, model, optimizer, epochs, batch_size, train_size, clip, test_kvasir_path, test_clinic_path) 
