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

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        class_ls = ['aeroplane','bicycle','boat','bus','car', 'chair','diningtable','motorbike','sofa','train']
        class_to_index = {'aeroplane': 0,'bicycle': 1,'boat': 2,'bus': 3,'car': 4,'chair': 5,'diningtable': 6,'motorbike': 7,'sofa': 8,'train': 9}
        val_nuisances = ['shape', 'pose', 'texture', 'context', 'weather','occlusion','iid']

        txt_dir = "./data/phase-1-pose-npz/train"
        img_dir = "./data/phase-1-pose-npz/train/processed/images"
        ann_dir = "./data/phase-1-pose-npz/train/processed/annotations/"
        label_list = []
        self.data_list = []
        for class_name in class_ls:
            with open(txt_dir+'/'+class_name +'.txt', "r", encoding='utf-8') as f:  #打开文本
                data = f.read()   #读取文本
            name_list = data.split('\n')
            label_list.append(name_list)
        for name in os.listdir(img_dir):
            img_path = img_dir+'/'+name
            label = class_to_index[class_name]
            for index in range(10):
                if name.split('.')[0] in label_list[index]:
                    label = index
                    break
            labels = np.zeros([1,10])
            labels[0,label]=1

            ann_file = ann_dir+ name.split('.')[0]+'.npz'
            data = np.load(ann_file,allow_pickle=True)

            self.data_list.append([img_path, labels.astype(np.float32),[(float(data['theta'])-theta_mean)/theta_std,(float(data['elevation'])-elevation_mean)/elevation_std,(float(data['distance'])-distance_mean)/distance_std,(float(data['azimuth'])-azimuth_mean)/azimuth_std]]) # img label 
        self.transform = transform
    def __getitem__(self, index):
        img_path, label, pose_value = self.data_list[index][0], self.data_list[index][1], self.data_list[index][2] # theta,elevation,distance,azimuth
        #print(img_path)
        image = cv2.imread(img_path)

        #print(image)
        image = cv2.resize(image, (224, 224),interpolation=cv2.INTER_CUBIC)

        image = self.transform(Image.fromarray(image))
        return image, label[0], np.array(pose_value).astype(np.float32)
    def __len__(self):
        return len(self.data_list)


class MyvailDataset(torch.utils.data.Dataset):
    def __init__(self, transform,ood_class):
        class_ls = ['aeroplane','bicycle','boat','bus','car', 'chair','diningtable','motorbike','sofa','train']
        class_to_index = {'aeroplane': 0,'bicycle': 1,'boat': 2,'bus': 3,'car': 4,'chair': 5,'diningtable': 6,'motorbike': 7,'sofa': 8,'train': 9}

        ann_dir = f"./data/phase-1-pose-npz/phase-1/nuisances/{ood_class}/annotations/"
        img_dir = f"./data/phase-1-pose-npz/phase-1/nuisances/{ood_class}/images"
        label_list = []
        self.data_list = []
        for name in os.listdir(img_dir):
            img_path = img_dir+'/'+name
            data = np.load(ann_dir + name.split('.')[0]+ '.npz',allow_pickle=True)
           
            label = class_to_index[name.split('_')[1]]
            labels = np.zeros([1,10])
            labels[0,label]=1
            self.data_list.append([img_path, labels.astype(np.float32),[(float(data['theta'])-theta_mean)/theta_std,(float(data['elevation'])-elevation_mean)/elevation_std,(float(data['distance'])-distance_mean)/distance_std,(float(data['azimuth'])-azimuth_mean)/azimuth_std]])
        self.transform = transform
    def __getitem__(self, index):
        img_path, label,pose_value = self.data_list[index][0], self.data_list[index][1], self.data_list[index][2]
        #print(img_path)
        image = cv2.imread(img_path)

        #print(image)
        image = cv2.resize(image, (224, 224),interpolation=cv2.INTER_CUBIC)

        image = self.transform(Image.fromarray(image))
        #print(label[0].shape)
        return image, label[0],np.array(pose_value ).astype(np.float32)
    def __len__(self):
        return len(self.data_list)

def mse_loss(outputs, target):
    return torch.nn.MSELoss(reduction='mean')(outputs,target)

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

def main():
        args = args_resnet50
        arch = 'resnet50'
        print(arch)
        assert args['epochs'] <= 200
        if args['batch_size'] > 256:
            # force the batch_size to 256, and scaling the lr
            args['optimizer_hyperparameters']['lr'] *= 256/args['batch_size']
            args['batch_size'] = 256
        # Data
        transform_train = transforms.Compose([
            transforms.RandomCrop(224, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_vail = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = MyDataset(transform=transform_train)

        val_loader_list = []
        val_nuisances = ['shape', 'pose', 'texture', 'context', 'weather','occlusion','iid']
        for val_name in val_nuisances :
            vailset = MyvailDataset(transform=transform_vail,ood_class = val_name)
            vailloader = data.DataLoader(vailset, batch_size=256, shuffle=False, num_workers=4)
            val_loader_list.append(vailloader)
        trainloader = data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
       

        model = timm.create_model('resnet50', pretrained=True, num_classes = 4)
        

        best_acc = 0  # best test accuracy
        vail_loss = 0
        vail_acc = 0
        optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
            **args['optimizer_hyperparameters'])
        if args['scheduler_name'] != None:
            scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
            **args['scheduler_hyperparameters'])

        model = nn.DataParallel(model, device_ids=[0,1,2,3])
        model = model.cuda()
        # Train and val
        logger.info('start training!')
        logger.info(args)
        print(args)

        thr = np.pi / 6
        gt_ood_dir = "./data/pose_ref/ref/nuisances"
        nuisances = ['shape', 'pose', 'texture', 'context', 'weather', 'occlusion','iid']
        gt_ood_list =[]
        for nuisance in nuisances:
            gt = pd.read_csv(gt_ood_dir + f'/{nuisance}/labels.csv')
            gt_ood_list.append(gt)
        for epoch in tqdm(range(args['epochs'])):

            train_loss = train(trainloader, model, optimizer)
            
            #print('acc: {}'.format(train_acc))

                
            vail_acc_list = []
            vail_loss_list = []
            
            for num in range(len(val_loader_list)):
                vailloader = val_loader_list[num]
                losses = AverageMeter()
                pred = pd.DataFrame()

                for (inputs, soft_labels,pose_value) in vailloader:
                    inputs, soft_labels,pose_value = inputs.cuda(), soft_labels.cuda(),pose_value.cuda()
                    outputs = model(inputs)
                    vail_loss = mse_loss(outputs, pose_value)
                    losses.update(vail_loss.item(), inputs.size(0))

                    outputs = outputs.detach().cpu().numpy()
                    pred = pd.concat([pred,pd.DataFrame(outputs)])

                pred.columns=['theta','elevation','distance','azimuth']  #theta,elevation,distance,azimuth
                pred['theta'] = (pred['theta'] * theta_std) + theta_mean
                pred['elevation'] = (pred['elevation'] * elevation_std) + elevation_mean
                pred['distance'] = (pred['distance'] * distance_std) + distance_mean
                pred['azimuth'] = (pred['azimuth'] * azimuth_std) + azimuth_mean
                vail_acc_list.append(get_acc(pred, gt_ood_list[num], thr))

                vail_loss = losses.avg
                vail_loss_list.append(vail_loss)
            vail_loss = np.mean(vail_loss_list[:6])
            vail_acc = np.mean(vail_acc_list[:6]) 
            logger.info('Epoch:[{}/{}]\t shape={:.3f}, pose={:.3f}, texture={:.3f}, context={:.3f}, weather={:.3f},occlusion={:.3f},iid={:.3f}'.format(epoch, args['epochs'],vail_acc_list[0],vail_acc_list[1],vail_acc_list[2],vail_acc_list[3],vail_acc_list[4],vail_acc_list[5],vail_acc_list[6]  ))

            logger.info('Epoch:[{}/{}]\t loss={:.5f}\t ood_loss={:.5f}\t ood_acc={:.3f} iid_acc={:.3f}'.format(epoch, args['epochs'], train_loss,vail_loss,vail_acc,vail_acc_list[6]))
        
            # save model
            if vail_acc>=best_acc:
                best_acc = max(vail_acc, best_acc)
                filepath = os.path.join('./output/'+arch + '.pth.tar')
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': vail_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, filepath)

        print('Best acc:')
        print(best_acc)
        logger.info('Best acc={:.3f}'.format(best_acc))
        logger.info('finish training!')

def train(trainloader, model, optimizer):
    losses = AverageMeter()
    model.eval()
    model.train()

    for (inputs, soft_labels,pose_value) in trainloader:
        inputs, soft_labels,pose_value = inputs.cuda(), soft_labels.cuda(),pose_value.cuda()

        outputs = model(inputs)
        loss = mse_loss(outputs, pose_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
    return losses.avg



if __name__ == '__main__':
    main()
