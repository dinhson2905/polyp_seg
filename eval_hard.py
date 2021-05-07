import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import configparser

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

def dice_coeff(inputs, targets, eps=1e-4):
    inputs_flat = np.reshape(inputs, -1)
    targets_flat = np.reshape(targets, -1)
    intersection = (inputs_flat * targets_flat)
    dice_score = (2 * intersection.sum() + eps) / (inputs_flat.sum() + targets_flat.sum() + eps)    
    return dice_score

def iou_coeff(inputs, targets, eps=1e-4):
    inputs_flat = np.reshape(inputs, -1)
    targets_flat = np.reshape(targets, -1)
    intersection = (inputs_flat * targets_flat)
    union = inputs_flat + targets_flat
    iou_score = (intersection.sum() + eps) / (union.sum() - intersection.sum() + eps)
    return iou_score


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    test_path = config['Paths']['test_path']
    # specify model_result_path
    eval_result_path = config['Paths']['eval_result_path']
    datasets = ['CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir', 'CVC-300']
    print('Model: ', eval_result_path)
    for data_name in datasets:
        image_root = f'{eval_result_path}/{data_name}/'
        gt_root = f'{test_path}/{data_name}/masks/'
        eval_loader = eval_dataset(image_root, gt_root)
        dice, iou = 0.0, 0.0
        for i in range(eval_loader.size):
            image, gt, name = eval_loader.load_data()
            gt = np.asarray(gt, np.float32)
            # gt /= (gt.max() + 1e-8)
            gt = (gt > 128)
            image = image
            inputs = image[0,1,:,:]
            inputs = (inputs > 0.5).float()
            inputs = np.array(inputs)
            targets = np.asarray(gt, np.float32)
            # print(inputs, targets)
            # print(inputs.sum(), targets.sum())
            dice_score = dice_coeff(inputs, targets)
            dice += dice_score
            iou_score = iou_coeff(inputs, targets)
            iou += iou_score
        
        mean_dice = dice / eval_loader.size
        mean_iou = iou / eval_loader.size
        print('{}: mean_dice: {:.3f}, mean_iou: {:.3f}'
                .format(data_name, mean_dice, mean_iou))
        