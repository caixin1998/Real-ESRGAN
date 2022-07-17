import random
import cv2
import math
import numpy as np
import os.path as osp
import os
import json
from scipy import rand
import torch
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,

                                               normalize)
import sys
sys.path.append("realesrgan")
from data.degrad_dataset import DegradDataset
from data.xgaze_dataset import XGazeDataset

@DATASET_REGISTRY.register()
class FullDataset(data.Dataset):
    def __init__(self, opt) -> None:
        super(FullDataset).__init__()
        self.datasets = []
        self.xgaze_dataset = XGazeDataset(opt)
        self.datasets.append(self.xgaze_dataset)
        self.gt_folders = opt["dataroot_gts"]
        for id in range(len(self.gt_folders)):
            self.datasets.append(DegradDataset(opt, dataset_id = id))
        self.dataset_num = len(self.datasets)
        self.len = 0
        for dataset in self.datasets:
            self.len = max(len(dataset), self.len)
    def __getitem__(self, index):
        i = random.randint(0,self.dataset_num - 1)
        if index < int(self.len / len(self.datasets[i])) * len(self.datasets[i]):
            return self.datasets[i][index % len(self.datasets[i])]
        else:
            return self.datasets[i][random.randint(0,len(self.datasets[i]) - 1)]
    def __len__(self):
        return self.len