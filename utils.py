import torch
import os
import torch.nn.functional as F
# import torchvision
import numpy as np
import argparse
import json
import pandas as pd
import os
import os.path as osp
import numpy as np
import math
from scipy.linalg import logm

def get_transformation_matrix(azimuth, elevation, distance):
    if distance == 0:
        # return None
        distance = 0.1

    # camera center
    C = np.zeros((3, 1))
    C[0] = distance * math.cos(elevation) * math.sin(azimuth)
    C[1] = -distance * math.cos(elevation) * math.cos(azimuth)
    C[2] = distance * math.sin(elevation)

    # rotate coordinate system by theta is equal to rotating the model by theta
    azimuth = -azimuth
    elevation = - (math.pi / 2 - elevation)

    # rotation matrix
    Rz = np.array([
        [math.cos(azimuth), -math.sin(azimuth), 0],
        [math.sin(azimuth), math.cos(azimuth), 0],
        [0, 0, 1],
    ])  # rotation by azimuth
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(elevation), -math.sin(elevation)],
        [0, math.sin(elevation), math.cos(elevation)],
    ])  # rotation by elevation
    R_rot = np.dot(Rx, Rz)
    R = np.hstack((R_rot, np.dot(-R_rot, C)))
    R = np.vstack((R, [0, 0, 0, 1]))

    return R

def rotation_theta(theta):
    # cos -sin  0
    # sin  cos  0
    # 0    0    1
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])


def cal_err(gt, pred):
    # return radius
    return ((logm(np.dot(np.transpose(pred), gt)) ** 2).sum()) ** 0.5 / (2. ** 0.5)


def cal_rotation_matrix(theta, elev, azum, dis):
    if dis <= 1e-10:
        dis = 0.5

    return rotation_theta(theta) @ get_transformation_matrix(azum, elev, dis)[0:3, 0:3]

def get_acc(pred, gt, thr):
    theta_anno, elevation_anno, azimuth_anno, distance_anno = [], [], [], []
    for idx, row in gt.iterrows():
        row.theta = row.theta * math.pi / 180
        row.elevation = row.elevation * math.pi / 180
        row.azimuth = row.azimuth * math.pi / 180
        # row.distance = row.distance * math.pi / 180
        theta_anno.append(row.theta)
        elevation_anno.append(row.elevation)
        azimuth_anno.append(row.azimuth)
        distance_anno.append(row.distance)

    theta_pred, elevation_pred, azimuth_pred, distance_pred = [], [], [], []
    for idx, row in pred.iterrows():
        row.theta = row.theta * math.pi / 180
        row.elevation = row.elevation * math.pi / 180
        row.azimuth = row.azimuth * math.pi / 180
        # row.distance = row.distance * math.pi / 180
        theta_pred.append(row.theta)
        elevation_pred.append(row.elevation)
        azimuth_pred.append(row.azimuth)
        distance_pred.append(row.distance)


    iid_error = []
    for theta_p, theta_a, elevation_p, elevation_a, azimuth_p, azimuth_a, distance_p, distance_a in zip(theta_pred, theta_anno, 
                                                                                                        elevation_pred, elevation_anno, 
                                                                                                        azimuth_pred, azimuth_anno, 
                                                                                                        distance_pred, distance_anno):
        anno_matrix = cal_rotation_matrix(theta_a, elevation_a, azimuth_a, distance_a)
        pred_matrix = cal_rotation_matrix(theta_p, elevation_p, azimuth_p, distance_p)
        if np.any(np.isnan(anno_matrix)) or np.any(np.isnan(pred_matrix)) or np.any(np.isinf(anno_matrix)) or np.any(np.isinf(pred_matrix)):
            error_ = np.pi / 2
        else:
            error_ = cal_err(anno_matrix, pred_matrix)
        iid_error.append(error_)
    iid_error = np.array(iid_error)
    
    acc = float(np.mean(iid_error < thr)) 
    return acc

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count