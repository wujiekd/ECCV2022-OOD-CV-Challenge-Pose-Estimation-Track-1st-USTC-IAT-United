# ECCV2022-OOD-CV-Challenge-Pose-Estimation-Track-USTC-IAT-United
Competition Open Source Solutions


## 1. Environment setting 

### 1.0. Package
* Several important packages
    - torch == 1.8.1+cu111
    - trochvision == 0.9.1+cu111

### 1.1. Dataset
In the classification track, we use only the OOD classification and detection data and labels:
* [ECCV-OOD](https://github.com/eccv22-ood-workshop/ROBIN-dataset)

### 1.2. OS
- [x] Windows10
- [x] Ubuntu20.04
- [x] macOS (CPU only)

## 2. Train
- [x] Single GPU Training
- [x] DataParallel (single machine multi-gpus)
- [ ] DistributedDataParallel

(more information: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### 2.1. data
train data and test data structure:  
```
├── data/
│   ├── OOD-CV-phase2/
│   │   ├── phase2-pose
│   │   ├── phase2-cls
│   │   └── phase2-det
│   ├── phase-1-pose-npz/
│   │   ├── phase1
│   │   └── train
│   ├── pose_ref/
│   │   ├── ref
└── └── └── res
```
  
Training sets and test sets are distributed with CSV labels corresponding to them.

### 2.2. run.
for details, see train.py
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
```

## 3. Evaluation
for details, see test.py
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py
```


### Acknowledgment

* Thanks to [timm](https://github.com/rwightman/pytorch-image-models) for Pytorch implementation.
