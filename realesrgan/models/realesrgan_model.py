import numpy as np
import random
import torch
import os
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from basicsr.archs import build_network
from basicsr.losses import build_loss
from torch.nn import functional as F
from torchvision.ops import roi_align


@MODEL_REGISTRY.register()
class RealESRGANModel(SRGANModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealESRGANModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening

        # train_opt = self.opt['train']
        # self.cri_l1 = build_loss(train_opt['L1_opt']).to(self.device)
        self.queue_size = opt.get('queue_size', 180)
        if 'network_gaze' in self.opt:
            self.network_gaze = build_network(self.opt["network_gaze"])
            self.network_gaze = self.model_to_device(self.network_gaze)
            self.print_network(self.network_gaze)
            load_path = self.opt['path'].get('pretrain_network_gaze')
            if load_path is not None:
                self.load_network(self.network_gaze, load_path, True, "state_dict")
            self.network_gaze.eval()
            for param in self.network_gaze.parameters():
                param.requires_grad = False

        if "network_uncertainty" in self.opt:
            load_path =self.opt["pretrain_network_uncertainty"]
            ckpt_path_list = []
            ckpt_list = os.listdir(load_path)
            for ckpt in ckpt_list:
                if ckpt[0:5] == "epoch":
                    if ckpt.endswith("%s.ckpt"%"mpii"):
                        ckpt_path_list += [os.path.join(load_path, ckpt)]
            ckpt_path_list.sort()
            pre_models = []
            start,end,interval = self.opt["sei"]
            for ckpt_path in ckpt_path_list[start:end:interval]:
                if os.path.isfile(ckpt_path):
                    pre_models.append(ckpt_path)

            self.network_uncertainty = []
            if pre_models is not None:
                for network_path in pre_models:
                    network = build_network(self.opt["network_uncertainty"])
                    network = self.model_to_device(network)
                    self.load_network(network, network_path, True, "state_dict")
                    network.eval()
                    for param in network.parameters():
                        param.requires_grad = False

                    self.network_uncertainty.append(network)



    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

        if 'loc_left_eye' in data:
                # get facial component locations, shape (batch, 4)
            self.loc_left_eyes = data['loc_left_eye']
            self.loc_right_eyes = data['loc_right_eye']
            # self.loc_mouths = data['loc_mouth']

        if 'gaze' in data:
            self.gaze = data['gaze'].to(self.device)

        if 'weight' in data:
            self.weight = data["weight"].to(self.device)
            self.weight = self.weight[:,None,None,None].expand(-1,*self.gt.shape[1:])
        else:
            self.weight = torch.ones(self.gts.shape).to(self.device)

    def get_roi_regions(self, eye_out_size=80):

        device = self.gt.device
        rois_eyes = []
        for b in range(self.loc_left_eyes.size(0)):  # loop for batch size
            # left eye and right eye
            img_inds = self.loc_left_eyes.new_full((2, 1), b)
            bbox = torch.stack([self.loc_left_eyes[b, :], self.loc_right_eyes[b, :]], dim=0)  # shape: (2, 4)
            rois = torch.cat([img_inds, bbox], dim=-1)  # shape: (2, 5)
            rois_eyes.append(rois)

        rois_eyes = torch.cat(rois_eyes, 0).to(device)

        # real images
        all_eyes = roi_align(self.gt, boxes=rois_eyes, output_size=eye_out_size)
        self.left_eyes_gt = all_eyes[0::2, :, :, :]
        self.right_eyes_gt = all_eyes[1::2, :, :, :]
        # output
        all_eyes = roi_align(self.output, boxes=rois_eyes, output_size=eye_out_size)
        self.left_eyes = all_eyes[0::2, :, :, :]
        self.right_eyes = all_eyes[1::2, :, :, :]

    def cri_uncertainty(self, x_in):
        pred_gaze_all = torch.zeros((x_in["face"].shape[0], len(self.network_uncertainty), 2))
        pred_gaze_all = pred_gaze_all.to(self.device)
        for i, network in enumerate(self.network_uncertainty):
            pred_gaze_all[:,i,:] = network(x_in)["pred"]
        uncertainty = torch.mean(torch.std(pred_gaze_all, dim = 1))
        return uncertainty

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealESRGANModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        self.output = self.weight * self.output + (1 - self.weight) * self.gt

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            if 'network_gaze' in self.opt:
                gaze_weight = self.opt['train']['gaze_weight']
                out_resize = self.output
                x_in = {"face":out_resize,
                "gaze":self.gaze}
                gaze_out = self.network_gaze(x_in)
                pred = torch.where(self.gaze != 0, gaze_out["pred"], self.gaze)
                l_gaze = self.cri_pix(pred, self.gaze) * gaze_weight
                l_g_total += l_gaze
                loss_dict['l_gaze'] = l_gaze

            if "network_uncertainty" in self.opt:
                uncertainty_weight = self.opt["train"]["uncertainty_weight"]
                x_in = {"face": self.output, "gaze": self.gaze}
                l_uncertainty = self.cri_uncertainty(x_in) * uncertainty_weight
                l_g_total += l_uncertainty
                loss_dict["l_uncertainty"] = l_uncertainty

            if "eye_weight" in self.opt['train']:
                self.get_roi_regions()
                eye_weight = self.opt['train']['eye_weight']
                l_g_eye = self.cri_pix(self.left_eyes, self.left_eyes_gt)
                r_g_eye = self.cri_pix(self.right_eyes, self.right_eyes_gt)
                loss_eye = eye_weight * (l_g_eye + r_g_eye)
                l_g_total += loss_eye

                loss_dict['l_g_eye'] = loss_eye

            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
