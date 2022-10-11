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
        
    



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--output', default='', type=str, metavar='PATH',
#                     help='path to the output dir')
# parser.add_argument('--input', default='./data/pose_ref', type=str, metavar='PATH',
#                     help='path to the input dir')
parser.add_argument('--iid-perf', default=0.787, type=float,
                    help='iid performance threshold')


theta_mean,elevation_mean,distance_mean,azimuth_mean= 0.00023595481126638268,0.11575889073578172,6.526204760750839,2.708895440086749
theta_std,elevation_std,distance_std,azimuth_std= 0.13191307591839532,0.1993462425530023,5.318982103694582,2.5564238212543766

def main():
    args = parser.parse_args()
    print(args)

    gt_ood_dir = "./data/pose_ref/ref/nuisances"

    pred_ood_dir = "./output/develop"

    thr = np.pi / 6
    gt_ood_dir = "./data/pose_ref/ref/nuisances"
    nuisances = ['shape', 'pose', 'texture', 'context', 'weather', 'occlusion','iid']
    accs = []
    i = 0
    for nuisance in nuisances:
        gt = pd.read_csv(gt_ood_dir + f'/{nuisance}/labels.csv')
        pred = pd.read_csv(pred_ood_dir+f'/{nuisance}.csv')
        pred['theta'] = theta_mean
        pred['elevation'] = elevation_mean
        pred['distance'] = distance_mean
        pred['azimuth'] = azimuth_mean
        accs.append(get_acc(pred, gt, thr))
        print(f"Acc@pi/6@{nuisance}: {accs[i]}")
        i+=1
    mean_acc = np.mean(accs[:6])
    print("Mean-Acc@pi/6: ", mean_acc)
    print("Current iid performance: ", accs[6])


if __name__ == '__main__':
    main()

    # import time
    # time.sleep(3 * 60)