# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training import lpips #import exportPerceptualLoss
import sys
# from unetseg.model import build_unet as unetseg
sys.path.append('.tmp/unetseg/UNET/')
from tmp.unetseg.UNET.model import build_unet as unetseg

import PIL
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        targets = torch.sigmoid(targets)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        targets = torch.sigmoid(targets)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE




class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.count_save_vessel = 0
        # print(device,type(device))
        self.percept = lpips.LPIPS(net='vgg').to(device)
        self.vesselseg = unetseg().to(device)
        #    save_pth = osp.join('./res/cp', model_name)
        self.vesselseg.load_state_dict(torch.load('./weights/unet.pth',map_location=device))
        self.vesselseg.eval()
        self.dicebce_loss = DiceBCELoss().to(self.device).eval()
        self.gen_vessel = None
        self.gt_vessel = None
        self.gen_img = None
        self.gt_img = None
    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        
        
        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
                ### perceptual_loss
                real_img_tmp = real_img.detach()
                perceptual_loss = self.percept(real_img_tmp, gen_img.detach())
                ### segment loss
                tmp_gen_img = gen_img.detach() ### (-1.0~1.0)
                tmp_gt_img = real_img.detach()
                maxv = torch.max(tmp_gen_img)
                minv = torch.min(tmp_gen_img)
                absmax = max(maxv,abs(minv))
                tmp_gen_img = tmp_gen_img/absmax  ## RGB->BGR (2,1,0)
                tmp_gen_img = (tmp_gen_img[:,[2,1,0],:,:]+1.0)*0.5  ### value:0~1.0
                tmp_gt_img = (tmp_gt_img[:,[2,1,0],:,:]+1.0)*0.5
                
                vessel_gen = self.vesselseg(tmp_gen_img) ## unetseg input should be (0~1)
                vessel_gt = self.vesselseg(tmp_gt_img)

                # parsing = out.squeeze().cpu().detach().numpy()
                # print(np.max(parsing),np.min(parsing),'parsing') ### (max:0.99,min:0.01)
                # mask = parsing > 0.5
                # print(np.sum(mask),mask.shape,' sum mask')
    

                seg_loss = self.dicebce_loss(vessel_gen,vessel_gt)
                self.gen_vessel = vessel_gen
                self.gt_vessel = vessel_gt
                self.gen_img = tmp_gen_img
                self.gt_img = tmp_gt_img
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (seg_loss+perceptual_loss+loss_Gmain).mean().mul(gain).backward()
                # (seg_loss+loss_Gmain).mean().mul(gain).backward()
                # loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
                ### perceptual_loss
                real_img_tmp = real_img.detach()
                perceptual_loss = self.percept(real_img_tmp, gen_img.detach())
                ### seg loss
                tmp_gen_img = gen_img.detach() ### (-1.0~1.0)
                tmp_gt_img = real_img.detach()
                maxv = torch.max(tmp_gen_img)
                minv = torch.min(tmp_gen_img)
                absmax = max(maxv,abs(minv))
                tmp_gen_img = tmp_gen_img/absmax  ## RGB->BGR (2,1,0)
                tmp_gen_img = (tmp_gen_img[:,[2,1,0],:,:]+1.0)*0.5  ### value:0~1.0，主要看算法生成的是rgb还是gbr
                tmp_gt_img = (tmp_gt_img[:,[2,1,0],:,:]+1.0)*0.5 ### 这里GT不变换即可
                
                vessel_gen = self.vesselseg(tmp_gen_img) ## unetseg input should be (0~1)，要看unetseg 接收的是RGB还是GBR
                vessel_gt = self.vesselseg(tmp_gt_img)
                seg_loss = self.dicebce_loss(vessel_gen,vessel_gt)
                self.gen_vessel = vessel_gen
                self.gt_vessel = vessel_gt
                self.gen_img = tmp_gen_img
                self.gt_img = tmp_gt_img
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (seg_loss+perceptual_loss+loss_Gpl).mean().mul(gain).backward()
                # (seg_loss+loss_Gpl).mean().mul(gain).backward()
                # loss_Gpl.mean().mul(gain).backward()

        if self.count_save_vessel % 10 == 1 and self.gen_vessel is not None:
            self.count_save_vessel = 1
            save_gen = torch.sigmoid(self.gen_vessel)
            save_gen = save_gen.permute(0, 2, 3, 1)
            save_gen = save_gen.squeeze().cpu().detach().numpy()
            save_gen = save_gen > 0.5
            save_gen = save_gen.astype('uint8')
            save_gt = torch.sigmoid(self.gt_vessel)
            save_gt = save_gt.permute(0, 2, 3, 1)
            save_gt = save_gt.squeeze().cpu().detach().numpy()
            save_gt = save_gt > 0.5  
            save_gt = save_gt.astype('uint8')
            
            gen_img = (self.gen_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            gen_img = gen_img.squeeze().cpu().detach().numpy()
            gt_img = (self.gt_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            gt_img = gt_img.squeeze().cpu().detach().numpy()
            
            
            C = self.gen_vessel.shape[1]
            fname = 'out/'
            if C == 1:
                PIL.Image.fromarray(save_gen[0]*255, 'L').save(fname+'gen_v.png')
                PIL.Image.fromarray(save_gt[0]*255, 'L').save(fname+'gt_v.png')
            if C == 3:
                PIL.Image.fromarray(save_gen[0]*255, 'RGB').save(fname+'gen_v.png')
                PIL.Image.fromarray(save_gt[0]*255, 'RGB').save(fname+'gt_v.png')
            PIL.Image.fromarray(gen_img[0], 'RGB').save(fname+'gen.png')
            PIL.Image.fromarray(gt_img[0], 'RGB').save(fname+'gt.png')
        self.count_save_vessel += 1
                
        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
