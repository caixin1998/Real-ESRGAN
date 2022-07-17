from basicsr.utils.registry import DATASET_REGISTRY
import torch.utils.data as data
import json
from basicsr.utils import img2tensor
# from data.image_folder import make_dataset
# from PIL import Image
import random, time
import os,h5py
import numpy as np
import torch
import cv2 as cv
import copy

import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch
import pathlib
import random
import torchvision.transforms as transforms

from torchvision.transforms.functional import normalize
from PIL import Image

@DATASET_REGISTRY.register()
class MPIIDataset(Dataset):

    def __init__(self, opt):
        self.lines = []
        self.pic_num = opt['dataroot_gt']
        self.root = opt['dataroot_gt']
        self.opt = opt
        self.split = opt["split"]
        self.crop_components = opt.get('crop_components', False)  # facial components
        self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)
        self.pic_num = opt.get('pic_num', 45000)
        self.label_path = os.path.join(self.root, "Label")
        self.im_root = os.path.join(self.root, "Image")
        # self.mean = opt['mean']
        # self.std = opt['std']
        person = opt.get('person', None)
        header = True
        self.path = [os.path.join(self.label_path, path) for path in os.listdir(self.label_path)]
        self.path.sort()
        if person is not None:
            self.path = self.path[person]

        if isinstance(self.path, list):
            for i in self.path:
                with open(i) as f:
                    line = f.readlines()
                    if header: line.pop(0)
                    self.lines.extend(line)
        else:
            with open(self.path) as f:
                self.lines = f.readlines()
                if header: self.lines.pop(0)
        random.shuffle(self.lines)
        if self.pic_num is not None:
            self.lines = self.lines[:self.pic_num]

    def get_components_bbox(self, lm):
        item_dict = {}
        map_left_eye = list(range(36, 42))
        map_right_eye = list(range(42, 48))
        map_mouth = list(range(48, 68))

        mean_left_eye = np.mean(lm[map_left_eye], 0)  # (x, y)
        half_len_left_eye = np.max((np.max(np.max(lm[map_left_eye], 0) - np.min(lm[map_left_eye], 0)) / 2, 16))
        item_dict['left_eye'] = [mean_left_eye[0], mean_left_eye[1], half_len_left_eye]
        # mean_left_eye[0] = 512 - mean_left_eye[0]  # for testing flip
        half_len_left_eye *= self.eye_enlarge_ratio

        # eye_right
        mean_right_eye = np.mean(lm[map_right_eye], 0)
        half_len_right_eye = np.max((np.max(np.max(lm[map_right_eye], 0) - np.min(lm[map_right_eye], 0)) / 2, 16))
        item_dict['right_eye'] = [mean_right_eye[0], mean_right_eye[1], half_len_right_eye]
        # mean_right_eye[0] = 512 - mean_right_eye[0]  # # for testing flip
        half_len_right_eye *= self.eye_enlarge_ratio

        # mouth
        mean_mouth = np.mean(lm[map_mouth], 0)
        half_len_mouth = np.max((np.max(np.max(lm[map_mouth], 0) - np.min(lm[map_mouth], 0)) / 2, 16))
        item_dict['mouth'] = [mean_mouth[0], mean_mouth[1], half_len_mouth]

        return item_dict

    def get_component_coordinates(self, lmks, status):
        """Get facial component (left_eye, right_eye, mouth) coordinates from a pre-loaded pth file"""

        components_bbox = self.get_components_bbox(lmks)
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
            components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

        # get coordinates
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations.append(loc)
        return locations

    def __len__(self):
        # if self.pic_num < 0:
        return len(self.lines)
        # return self.pic_num

    def __getitem__(self, idx):

        line = self.lines[idx]
        line = line.strip().split(" ")
        # print(line)

        # name = line[0].split('/')[0]
        gaze2d = line[1]
        head2d = line[2]

        face_path = line[0]
        lmks = np.array(line[3].split(",")).astype("float").reshape(68, 2)

        label = np.array(gaze2d.split(",")).astype("float")

        label = torch.from_numpy(label).type(torch.FloatTensor)
        headpose = np.array(head2d.split(",")).astype("float")
        headpose = torch.from_numpy(headpose).type(torch.FloatTensor)

        face = cv2.imread(os.path.join(self.im_root, face_path))
        face = cv2.resize(face, (512, 512))
        lmks =lmks * 512. / 224.
        gt_path = os.path.join(self.im_root, face_path)
        img_gt = face.astype(np.float32) / 255.
        img_lq = img_gt.copy()
        img_lq = cv2.resize(img_lq, (256, 256))
        # img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]

        # normalize(img_gt, self.mean, self.std, inplace=True)
        # normalize(img_lq, self.mean, self.std, inplace=True)
        # print(img_gt.shape)

        if self.crop_components:
            locations = self.get_component_coordinates(lmks, [0])
            loc_left_eye, loc_right_eye, loc_mouth = locations
        # if self.opt.half == 1:
        #     y = np.concatenate((lmk[42:47],lmk[36:41]))[:,1].mean()
        #     # face[int(y - 10): int(y + 10), ...] = 0
        #     if 168 > y > 56:
        #         face = face[int(y - 56):
        #         int(y - 56) + 112, ...]
        #     else:
        #         face = face[:112,...]

        # if self.opt.half == 2:
        #     y = np.concatenate((lmk[42:47],lmk[36:41]))[:,1].mean()
        #     if 192 > y > 28:
        #         face = face[int(y - 28):
        #         int(y - 28) + 56, ...]
        #     else:
        #         face = face[:56,...]

        # ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        # ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        # face = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


        # face = self.transform(face)
        if self.crop_components:
            return_dict = {
                'lq': img_lq,
                'gt': img_gt,
                'gt_path': gt_path,
                'lq_path': gt_path,
                'loc_left_eye': loc_left_eye,
                'loc_right_eye': loc_right_eye,
                'loc_mouth': loc_mouth,
                'gaze': label
            }

        else:
            return_dict = {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path, 'lq_path': gt_path,'gaze': label}

        # print(os.path.join(self.im_root, face_path), label)
        # img = {"left":torch.from_numpy(limg).type(torch.FloatTensor),
        #        "right":torch.from_numpy(rimg).type(torch.FloatTensor),
        #        "face":torch.from_numpy(fimg).type(torch.FloatTensor),
        #        "head_pose":headpose,
        #        "name":name}

        return return_dict




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # seed_everything(1)
    path = '/home/caixin/GazeData/MPIIFaceGaze/Label/p00.label'
    d = txtload(path, '/home/caixin/GazeData/MPIIFaceGaze/Image', batch_size=32, pic_num=5,
                shuffle=False, num_workers=4, header=True)
    print(len(d))
    for i, (img, label) in enumerate(d):
        print(i, label)
