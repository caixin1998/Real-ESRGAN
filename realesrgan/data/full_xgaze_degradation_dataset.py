

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
@DATASET_REGISTRY.register()
class FullXGazeDegradationDataset(data.Dataset):
    def __init__(self, opt) -> None:
        super(FullXGazeDegradationDataset).__init__()
        self.datasets = []
        self.xgaze_dataset = XGazeDegradationDataset(opt)
        self.datasets.append(self.xgaze_dataset)
        self.gt_folders = opt["dataroot_gts"]
        for id in range(len(self.gt_folders)):
            self.datasets.append(DegradationDataset(opt, dataset_id = id))
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


MATCHED_PARTS = ([1, 17], [2, 16], [3, 15], [4, 14], [5, 13], [6, 12], [7, 11], [8, 10],
             [18, 27], [19, 26], [20, 25], [21, 24], [22, 23],
             [32, 36], [33, 35],
             [37, 46], [38, 45], [39, 44], [40, 43], [41, 48], [42, 47],
             [49, 55], [50, 54], [51, 53], [62, 64], [61, 65], [68, 66], [59, 57], [60, 56])

def fliplr_joints(x, width):
    x[:, 0] = width - x[:, 0]
    for pair in MATCHED_PARTS:
        tmp = x[pair[0] - 1, :].copy()
        x[pair[0] - 1, :] = x[pair[1] - 1, :]
        x[pair[1] - 1, :] = tmp
    return x

def generate_target(img, pt, sigma, label_type='Gaussian'):
# Check that any part of the gaussian is in-bounds
    tmp_size = sigma * 3
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if label_type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

class XGazeDegradationDataset(data.Dataset):
    """XGaze dataset for GFPGAN.

    It reads high resolution images, and then generate low-quality (LQ) images on-the-fly.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(XGazeDegradationDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        # self.io_backend_opt = opt['io_backend']
        self.split = opt["split"]

        def read_json(refer_list_file):
            with open(refer_list_file, 'r') as f:
                datastore = json.load(f)
            return datastore

        self.key_to_use = read_json(os.path.join(opt['dataroot_gt_xgaze'], "train_valid_split.json"))[self.split]

        self.gt_folder = os.path.join(opt['dataroot_gt_xgaze'], "train")
        self.mean = opt['mean']
        self.std = opt['std']
        self.out_size = opt['out_size']
        self.root = self.gt_folder
        self.label_path = os.path.join(self.root, "Label")
        self.im_root = os.path.join(self.root, "Image")
        self.use_lmk = opt.get('use_lmk', False)
        self.crop_components = opt.get('crop_components', False)  # facial components
        self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)  # whether enlarge eye regions



        # file client (lmdb io backend)
        self.path = [osp.join(self.label_path, path) for path in os.listdir(self.label_path) if path.split('.')[0][-4:] in self.key_to_use]
        self.path.sort()
        self.lines = []

        if isinstance(self.path, list):
            for i in self.path:
                with open(i) as f:
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(self.path) as f:
                self.lines = f.readlines()
                self.lines.pop(0)
        self.selected_lines = []
        for i in range(0,11):
            self.selected_lines += self.lines[i::18]
        self.selected_lines += self.lines[17::18]
        self.lines = self.selected_lines
        # degradation configurations
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.downsample_range = opt['downsample_range']
        self.noise_range = opt['noise_range']
        self.jpeg_range = opt['jpeg_range']

        # color jitter
        self.color_jitter_prob = opt.get('color_jitter_prob')
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob')
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        # to gray
        self.gray_prob = opt.get('gray_prob')

        logger = get_root_logger()
        logger.info(f'Blur: blur_kernel_size {self.blur_kernel_size}, sigma: [{", ".join(map(str, self.blur_sigma))}]')
        logger.info(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
        logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
        logger.info(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        if self.color_jitter_prob is not None:
            logger.info(f'Use random color jitter. Prob: {self.color_jitter_prob}, shift: {self.color_jitter_shift}')
        if self.gray_prob is not None:
            logger.info(f'Use random gray. Prob: {self.gray_prob}')
        self.color_jitter_shift /= 255.

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

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
        # mean_mouth[0] = 512 - mean_mouth[0]  # for testing flip


    def get_component_coordinates(self, lmks):
        """Get facial component (left_eye, right_eye, mouth) coordinates from a pre-loaded pth file"""

        components_bbox = self.get_components_bbox(lmks)
        # if status[0]:  # hflip
        #     # exchange right and left eye
        #     tmp = components_bbox['left_eye']
        #     components_bbox['left_eye'] = components_bbox['right_eye']
        #     components_bbox['right_eye'] = tmp
        #     # modify the width coordinate
        #     components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
        #     components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
        #     components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

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

    def __getitem__(self, index):
        # if self.file_client is None:
        #     self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.

        line = self.lines[index]
        line = line.strip().split(" ")

        gaze2d = line[1]
        head2d = line[2]
        face_path = line[0]
        gt_path = face_path
        lmks = np.array(line[3].split(",")).astype("float").reshape(68, 2)

        label = np.array(gaze2d.split(",")).astype("float")
        # print("label", label)
        headpose = np.array(head2d.split(",")).astype("float")
        headpose = torch.from_numpy(headpose).type(torch.FloatTensor)

        img_gt = cv2.imread(os.path.join(self.im_root, face_path))
        img_gt = img_gt.astype(np.float32) / 255.
        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)

        if status[0]:
            lmks = fliplr_joints(lmks, self.out_size)

        eye_lmks = lmks[36:48]
        nparts = eye_lmks.shape[0]
        h, w, _ = img_gt.shape
        target = np.zeros((nparts, h, w))
        tpts = eye_lmks.copy()
        if self.use_lmk:
            for i in range(nparts):
                if tpts[i, 1] >= 0:
                    target[i] = generate_target(target[i], tpts[i], 1.5,
                                                label_type='Gaussian')
        # get facial component coordinates
        if self.crop_components:
            locations = self.get_component_coordinates(lmks)
            loc_left_eye, loc_right_eye, loc_mouth = locations

        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            if self.opt.get('gt_gray'):  # whether convert GT to gray images
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])  # repeat the color channels

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # random color jitter (pytorch version) (only for lq)
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            img_lq = self.color_jitter_pt(img_lq, brightness, contrast, saturation, hue)

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        normalize(img_gt, self.mean, self.std, inplace=True)
        normalize(img_lq, self.mean, self.std, inplace=True)

        if status[0]:
            label = np.array([label[0], -label[1]], dtype=np.float32)
        label = torch.from_numpy(label).type(torch.FloatTensor)
        target = torch.FloatTensor(target)
        if self.crop_components:
            return_dict = {
                'lq': img_lq,
                'gt': img_gt,
                'gt_path': gt_path,
                'loc_left_eye': loc_left_eye,
                'loc_right_eye': loc_right_eye,
                'loc_mouth': loc_mouth,
                'gaze': label,
                'target': target
            }
            return return_dict
        else:
            return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path, 'gaze': label, 'target': target}

    def __len__(self):
        return len(self.lines)


class DegradationDataset(data.Dataset):
    """FFHQ dataset for GFPGAN.

    It reads high resolution images, and then generate low-quality (LQ) images on-the-fly.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
            Please see more options in the codes.
    """

    def __init__(self, opt, dataset_id):
        super(DegradationDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gts'][dataset_id]
        self.mean = opt['mean']
        self.std = opt['std']
        self.out_size = opt['out_size']

        self.use_lmk = opt.get('use_lmk', False)
        self.crop_components = opt.get('crop_components', False)  # facial components
        self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)  # whether enlarge eye regions

        # if self.crop_components:
            # load component list from a pre-process pth files
            # self.components_list = torch.load(opt.get('component_path'))

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # disk backend: scan file list from a folder
            # self.paths = paths_from_folder(self.gt_folder)
            # self.paths.sort()
            self.paths = []
            with open(os.path.join(self.gt_folder, "lmk.Label")) as f:
                self.paths = f.readlines()
                self.paths.pop(0)
        # degradation configurations
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.downsample_range = opt['downsample_range']
        self.noise_range = opt['noise_range']
        self.jpeg_range = opt['jpeg_range']

        # color jitter
        self.color_jitter_prob = opt.get('color_jitter_prob')
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob')
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        # to gray
        self.gray_prob = opt.get('gray_prob')

        logger = get_root_logger()
        logger.info(f'Blur: blur_kernel_size {self.blur_kernel_size}, sigma: [{", ".join(map(str, self.blur_sigma))}]')
        logger.info(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
        logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
        logger.info(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        if self.color_jitter_prob is not None:
            logger.info(f'Use random color jitter. Prob: {self.color_jitter_prob}, shift: {self.color_jitter_shift}')
        if self.gray_prob is not None:
            logger.info(f'Use random gray. Prob: {self.gray_prob}')
        self.color_jitter_shift /= 255.

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

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
        # mean_mouth[0] = 512 - mean_mouth[0]  # for testing flip


    def get_component_coordinates(self, lmks):
        """Get facial component (left_eye, right_eye, mouth) coordinates from a pre-loaded pth file"""

        components_bbox = self.get_components_bbox(lmks)
        # if status[0]:  # hflip
        #     # exchange right and left eye
        #     tmp = components_bbox['left_eye']
        #     components_bbox['left_eye'] = components_bbox['right_eye']
        #     components_bbox['right_eye'] = tmp
        #     # modify the width coordinate
        #     components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
        #     components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
        #     components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

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

    def __getitem__(self, index):
        # if self.file_client is None:
        #     self.file_client = FileClient("disk", **self.io_backend_opt)

        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = os.path.join(self.gt_folder, self.paths[index].strip().split(" ")[0])
        lmks = np.array(self.paths[index].strip().split(" ")[1].split(",")).astype("float").reshape(68, 2)
        # gt_path =self.paths[index]

        img_gt = cv2.imread(gt_path)
        img_gt = img_gt.astype(np.float32) / 255.

        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)


        if status[0]:
            lmks = fliplr_joints(lmks, self.out_size)
        h, w, _ = img_gt.shape

        eye_lmks = lmks[36:48]
        nparts = eye_lmks.shape[0]
        target = np.zeros((nparts, h, w))
        tpts = eye_lmks.copy()
        if self.use_lmk:
            for i in range(nparts):
                if tpts[i, 1] >= 0:
                    target[i] = generate_target(target[i], tpts[i], 1.5,
                                                label_type='Gaussian')

        # get facial component coordinates
        if self.crop_components:
            locations = self.get_component_coordinates(lmks)
            loc_left_eye, loc_right_eye, loc_mouth = locations

        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            if self.opt.get('gt_gray'):  # whether convert GT to gray images
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])  # repeat the color channels

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # random color jitter (pytorch version) (only for lq)
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            img_lq = self.color_jitter_pt(img_lq, brightness, contrast, saturation, hue)

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        normalize(img_gt, self.mean, self.std, inplace=True)
        normalize(img_lq, self.mean, self.std, inplace=True)
        target = torch.FloatTensor(target)

        if self.crop_components:
            return_dict = {
                'lq': img_lq,
                'gt': img_gt,
                'gt_path': gt_path,
                'loc_left_eye': loc_left_eye,
                'loc_right_eye': loc_right_eye,
                'loc_mouth': loc_mouth,
                'gaze': torch.zeros((2)),
                'target': target
            }
            return return_dict
        else:
            return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path, 'gaze': torch.zeros((2)), 'target': target}

    def __len__(self):
        return len(self.paths)