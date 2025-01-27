import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from realesrgan.data.utils import get_component_coordinates



# @DATASET_REGISTRY.register()

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

class XGazeDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(XGazeDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt_xgaze']
        self.split = opt["split"]
        self.out_size = 512
        def read_json(refer_list_file):
            import json
            with open(refer_list_file, 'r') as f:
                datastore = json.load(f)
            return datastore
        # file client (lmdb io backend)

            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
        self.key_to_use = read_json(os.path.join(opt['dataroot_gt_xgaze'], "train_valid_split.json"))[self.split]

        self.gt_folder = os.path.join(opt['dataroot_gt_xgaze'], "train")

        self.root = self.gt_folder
        self.label_path = os.path.join(self.root, "Label")
        self.im_root = os.path.join(self.root, "Image")

        self.use_lmk = opt.get('use_lmk', False)
        self.crop_components = opt.get('crop_components', True)  # facial components
        self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)  # whether enlarge eye regions

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

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1


    def __getitem__(self, index):

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        line = self.lines[index]
        line = line.strip().split(" ")
        gaze2d = line[1]
        head2d = line[2]
        face_path = line[0]
        gt_path = face_path
        if self.crop_components:
            lmks = np.array(line[3].split(",")).astype("float").reshape(68, 2)
            locations = get_component_coordinates(lmks)
            loc_left_eye, loc_right_eye = locations

        label = np.array(gaze2d.split(",")).astype("float")
        # print("label", label)
        headpose = np.array(head2d.split(",")).astype("float")
        headpose = torch.from_numpy(headpose).type(torch.FloatTensor)

        img_gt = cv2.imread(os.path.join(self.im_root, face_path))
        img_gt = img_gt.astype(np.float32) / 255.
        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt, status = augment(img_gt, self.opt['use_hflip'], rotation=False, return_status=True)

        if status[0] and self.crop_components:
            lmks = fliplr_joints(lmks, self.out_size)

        # eye_lmks = lmks[36:48]
        # nparts = eye_lmks.shape[0]
        # h, w, _ = img_gt.shape
        # target = np.zeros((nparts, h, w))
        # tpts = eye_lmks.copy()
        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        # h, w = img_gt.shape[0:2]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        if status[0]:
            label = np.array([label[0], -label[1]], dtype=np.float32)
        label = torch.from_numpy(label).type(torch.FloatTensor)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path, 'gaze': label, 'weight': 0}
        if self.crop_components:
            return_d["loc_left_eye"] = loc_left_eye
            return_d["loc_right_eye"] = loc_right_eye

        return return_d

    def __len__(self):
        return len(self.lines)
