import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import configparser
from statistics import mean
import logging
from logs import setup_logger

class eval_dataset:
    def __init__(self, image_root, gt_root):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.ToTensor()
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


def Fmeasure_calu(resmap, gt, gtsize,  threshold):
    if threshold > 1:
        threshold = 1
    
    label3 = np.zeros(gtsize)
    label3[resmap >= threshold] = 1
    num_rec = len(label3[np.nonzero(label3 == 1)])
    num_no_rec = len(label3[np.nonzero(label3 == 0)])
    label_and = np.logical_and(label3, gt)
    TP = len(label_and[np.nonzero(label_and == 1)])
    num_obj = gt.sum()
    num_pred = label3.sum()
    FN = num_obj - TP
    FP = num_rec - TP
    TN = num_no_rec - FN

    if TP == 0:
        dice = 0
        iou = 0
        pre = 0
        recall = 0
        specif = 0
        fmeasure = 0
    else:
        iou = TP / (FN + num_rec)
        pre = TP / num_rec
        recall = TP / num_obj
        dice = 2 * TP / (num_obj + num_pred)
        specif = TN / (TN + FP)
        fmeasure = ((2.0 * pre  * recall) / (pre + recall)) # beta = 1.0

    return dice, iou

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    test_path = config['Paths']['test_path']
    result_path = 'results/'
    # models = ['HarDPD', 'HarDCPD', 'PraHarDNet', 'ResNetPD', 'ResNetCPD', 'PraNet', 'UNet', 'UNet++']
    models = ['UNet']
    datasets = ['CVC-ClinicDB', 'ETIS-LaribPolypDB', 'CVC-300', 'Kvasir', 'CVC-ColonDB']
    # datasets = ['CVC-ClinicDB']
    for model in models:
        log_file = 'logs/eval_' + model + '.log'
        eval_logger = setup_logger('eval_logger', log_file)
        print('Model: ', model)
        eval_logger.info(model)
        eval_result_path = result_path + model
        for data_name in datasets:
            eval_logger.info(data_name)
            img_root = f'{eval_result_path}/{data_name}/'
            gt_root = f'{test_path}/{data_name}/masks/'
            eval_loader = eval_dataset(img_root, gt_root)
            thresholds = [0.5]            
            threshold_dice, threshold_IoU = np.zeros((eval_loader.size, len(thresholds))), np.zeros((eval_loader.size, len(thresholds)))
            
            for i in range(eval_loader.size):
                image, gt, name = eval_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt = (gt > 128)
                dgt = np.asarray(gt, np.float) # double_gt    
                resmap = image[0,1,:,:]
                resmap = np.array(resmap)
                threshold_dic, threshold_iou = np.zeros(len(thresholds)), np.zeros(len(thresholds))
                for j in range(0, len(thresholds)):
                    threshold_dic[j], threshold_iou[j] = Fmeasure_calu(resmap, dgt, gt.shape, thresholds[j])
                    eval_logger.info(f'{name} -  {threshold_dic[j]}')
                threshold_dice[i,:] = threshold_dic
                threshold_IoU[i,:] = threshold_iou
            # Dice
            column_dice = np.mean(threshold_dice, 0)        
            meandice = np.mean(column_dice)
            # IoU
            column_iou = np.mean(threshold_IoU, 0)    
            meaniou = np.mean(column_iou)
            print(meandice, meaniou)    