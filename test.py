from __future__ import print_function

import os
import random
import shutil
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
from PIL import Image
from config import args_resnet50
from utils import AverageMeter, accuracy
import logging
import sys
import timm
from evaluate_pose import get_acc

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# Use CUDA

logger = get_logger('./exp.log')
# Use CUDA
use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

theta_mean,elevation_mean,distance_mean,azimuth_mean= 0.00023595481126638268,0.11575889073578172,6.526204760750839,2.708895440086749
theta_std,elevation_std,distance_std,azimuth_std= 0.13191307591839532,0.1993462425530023,5.318982103694582,2.5564238212543766



class MytestDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        img_dir = f"./data/OOD-CV-phase2/phase2-pose/images"
        self.data_list = []
        for name in os.listdir(img_dir):
            img_path = img_dir+'/'+name
            self.data_list.append([img_path])
        self.transform = transform
    def __getitem__(self, index):
        img_path = self.data_list[index][0]
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224),interpolation=cv2.INTER_CUBIC)

        image = self.transform(Image.fromarray(image))
        return image,img_path
    def __len__(self):
        return len(self.data_list)

def mse_loss(outputs, target):
    return torch.nn.MSELoss(reduction='mean')(outputs,target)

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

def main():
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = MytestDataset(transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)
       
        model = timm.create_model('resnet50', pretrained=True, num_classes = 4,checkpoint_path='./output/resnet50.pth.tar')
 
        
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
        model = model.cuda()
        pred = pd.DataFrame()
        img_list = []
        for (inputs, img_path) in testloader:
                inputs = inputs.cuda()
                outputs = model(inputs)
                outputs = outputs.detach().cpu().numpy()
                pred = pd.concat([pred,pd.DataFrame(outputs)])
                img_list.extend(img_path)
              
             

        pred.columns=['theta','elevation','distance','azimuth']  #theta,elevation,distance,azimuth
        pred['theta'] = (pred['theta'] * theta_std) + theta_mean
        pred['elevation'] = (pred['elevation'] * elevation_std) + elevation_mean
        pred['distance'] = (pred['distance'] * distance_std) + distance_mean
        pred['azimuth'] = (pred['azimuth'] * azimuth_std) + azimuth_mean
        pred['imgs'] = [img.split('/')[-1] for img in img_list]
        pred['labels'] = [img.split('/')[-1].split('.')[0].split('_')[-1] for img in img_list]

        pred.to_csv('output/pred.csv',index=None)

if __name__ == '__main__':
    # print(sys.argv[1])
    main()
