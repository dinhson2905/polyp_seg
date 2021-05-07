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
from libs.hardmseg.hardmseg import HarDMSEG
from losses.region_loss import WIoUBCELoss
import configparser


train_logger = setup_logger('train_logger', 'logs/train.log')

def struct_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2,3)) / weit.sum(dim=(2,3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2,3))
    union = ((pred + mask) * weit).sum(dim=(2,3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def validation(model, path):
    data_path = path
    model.eval()
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = ValidationDataset(image_root, gt_root, 352)
    b = 0.0
    for i in range(100):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        _, res = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=True)
        res = res.sigmoid().data.cpu().numpy().squeeze()

        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)

        loss = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        a = '{:4f}'.format(loss)
        a = float(a)
        b = b + a
    return b / 100

def train(train_loader, model, optimizer, epoch, test_path, best):
    model.train()
    size_rates = [0.75, 1, 1.25]
    loss_attention_record, loss_detection_record = AvgMeter(), AvgMeter()
    criterion = WIoUBCELoss()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            trainsize = int(round(opt.trainsize * rate / 32) *32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
            attention_map, detection_map = model(images)
            loss1 = criterion(attention_map, gts)
            loss2 = criterion(detection_map, gts)
            loss = loss1 + loss2
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1:
                loss_attention_record.update(loss1.data, opt.batchsize)
                loss_detection_record.update(loss2.data, opt.batchsize)
        
        if i%20 == 0 or i == total_step:
            print(f'{datetime.now()} Epoch [{epoch}/{opt.epoch}], Step [{i}/{total_step}], \
                    Loss [attention_loss: {loss_attention_record.show()}, detection_loss: {loss_detection_record.show()}]')
            train_logger.info(f'{datetime.now()} Epoch [{epoch}/{opt.epoch}], Step [{i}/{total_step}], \
                                Loss [attention_loss: {loss_attention_record.show()}, detection_loss: {loss_detection_record.show()}]')
    
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % 1 == 0:
        meandice = validation(model, test_path)
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path + 'HarDMSEG-best.pth')
            print('[Saving Snapshots:]', save_path + 'HarDMSEG-best.pth', meandice)
            train_logger.info(f'[Saving Snapshots: {save_path + "HarGMSEG-best"} {meandice}]')
    
    return best

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='choosing optimizer Adam or SGD')
    parser.add_argument('--augmentation', default=False, help='choosen to do random flip rotation')
    parser.add_argument('--batchsize', type=int, default=16, help='training batchsize')
    parser.add_argument('--trainsize', type=int, default=352, help='training batchsize')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str, default='data/TrainDataset/', help='path to dataset')
    parser.add_argument('--test_path', type=str, default='data/TestDataset/Kvasir', help='path to test dataset')
    parser.add_argument('--train_save', type=str, default='hardmseg')

    opt = parser.parse_args()

    model = HarDMSEG().cuda()

    params = model.parameters()

    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)
    
    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation=opt.augmentation)
    total_step = len(train_loader)

    print('#'*20, 'Start Training', '#'*20)
    best = 0
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        best = train(train_loader, model, optimizer, epoch, opt.test_path, best) 
