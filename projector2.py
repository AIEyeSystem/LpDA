# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

### https://github.com/ouhenio/stylegan3-projector
### 总是被free的某个原因可能是batch中我只取了一个,或者某些变量放在了循环之外，输入也许并不需要grad，只有参数需要grad
### 问题应该是vessel_gt,和vessl_lesion不要grad，因为在loop外面

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter
import sys

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torch.nn as nn

from torchvision import datasets, models, transforms
    
from torch.utils.data import DataLoader, Dataset
import argparse

import matplotlib.pyplot as plt
from PIL import Image

# sys.path.append('.tmp/')
from tmp.unetseg.model import build_unet as unetseg
unetsegpth = 'tmp/unetseg/checkpoint.pth'

import dnnlib
import legacy

######################################## vessel loss ##############################################
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
def vessel_loss_init(device):
        vesselseg = unetseg()
        vesselseg.load_state_dict(torch.load(unetsegpth,map_location=device))
        for name,param in vesselseg.named_parameters():
            param.requires_grad = False
        # vesselseg = vesselseg
        # dicebce_loss = DiceBCELoss().to(device).eval()
        dicebce_loss = DiceBCELoss()
        return vesselseg,dicebce_loss
##############    
########################################################################################
############################  lesion segment  ##########################################
########################################################################################
# def lesion_pre_processing(img,device):
#     import cv2
#     cliplimit = 2
#     gridsize = 8
#     image = img.clone().squeeze(0)
#     image = image.cpu().numpy()
#     # print(image.shape,'image shape.....')
    

#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     black_mask = np.uint8((image_gray > 15)*255.)
#     ret, thresh = cv2.threshold(black_mask, 127, 255, 0)
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     mask = np.ones(image.shape[:2], dtype='uint8')*255
#     cn = []
#     for contour in contours:
#         if len(contour) > len(cn):
#             cn = contour
#     cv2.drawContours(mask, [cn], -1, 0, -1)
#     ## mask
    
    
#     # brightness balance.
#     brightnessbalance = False
#     if brightnessbalance:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         mask_img = mask
#         brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum()/255.)
#         image = np.uint8(np.minimum(image * brightnessbalance / brightness, 255))

#     # illumination correction and contrast enhancement.
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     lab_planes = list(cv2.split(lab))
#     clahe = cv2.createCLAHE(clipLimit=cliplimit,tileGridSize=(gridsize,gridsize))
#     lab_planes[0] = clahe.apply(lab_planes[0])
#     lab = cv2.merge(lab_planes)
#     nimg = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
#     denoise = True
#     if denoise:
#         nimg = cv2.fastNlMeansDenoisingColored(nimg, None, 10, 10, 1, 3)
#         nimg = cv2.bilateralFilter(nimg, 5, 1, 1)

    
#     # plt.subplot(223)
#     # plt.imshow(nimg)
#     nimg = torch.from_numpy(nimg).unsqueeze(0).to(device)
#     return nimg

sys.path.insert(0,'./ThirdPart/DR-segmentation/HEDNet_cGAN/')
# print(sys.path)
from transform.transforms_group import *
def get_lesion_mask(image,model,device):
    # print(image.size())
    image = image[:,[2,1,0],:,:]
    # print('in getLesion mask,',image.size())
    import config_gan_ex as config
    image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)#.to(torch.uint8)




    image_size = config.IMAGE_SIZE
    image_dir = config.IMAGE_DIR

    softmax = nn.Softmax(1)

    def eval_model(model, image,device):
        model.eval()
        masks_soft = []
        masks_hard = []
        m_transform = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        # transform=Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # with torch.set_grad_enabled(False):
        if True:
            inputs = image.permute(0,3,1,2)
            maxv = torch.max(inputs)
            minv = torch.min(inputs)
            absmax = max(maxv,abs(minv))
            inputs = inputs/absmax  ## 
            # print(torch.max(img),torch.min(img),' max min img for lesion seg')
            ## (0~255)->(0~1)
            # inputs = (inputs-0.5)*2.0
            inputs = m_transform(inputs)
            ### torch.Size([1, 3, 256, 256]) tensor(1., device='cuda:0') tensor(-1., device='cuda:0')
            inputs = inputs.to(device=device, dtype=torch.float)
            bs, _, h, w = inputs.shape
            # not ignore the last few patches
            h_size = (h - 1) // image_size + 1
            w_size = (w - 1) // image_size + 1
            masks_pred = torch.zeros(inputs.shape).to(dtype=torch.float)[:,:2,:,:]
            # print(h_size,w_size,'h_w_size')
            for i in range(h_size):
                for j in range(w_size):
                    h_max = min(h, (i + 1) * image_size)
                    w_max = min(w, (j + 1) * image_size)

                    inputs_part = inputs[:,:, i*image_size:h_max, j*image_size:w_max]
                    # inputs_part = inputs_part
                    # print('input_part size: ',inputs_part.size(),torch.max(inputs_part),torch.min(inputs_part))
                    masks_pred_single = model(inputs_part)[-1]
                    # print('masks_pred_single: ',masks_pred_single.size(),masks_pred.size())
                    masks_pred[:, :, i*image_size:h_max, j*image_size:w_max] = masks_pred_single
                    # plt.subplot(h_size,w_size,(i+1)*(j+1))
                    # plt.imshow(masks_pred_single.cpu().numpy()[0][0])
                    ## 分块分割
            masks_pred_softmax_batch = softmax(masks_pred)
            masks_soft_batch = masks_pred_softmax_batch[:, 1:, :, :]

        return masks_soft_batch[0][0]


    
    # print('before preprocessing : ',image.size(),torch.max(image),torch.min(image))
    # pre_image = lesion_pre_processing(image,device)
    pre_image = image #lesion_pre_processing(image,device)
    # print('after preprocessing : ',pre_image.size(),torch.max(pre_image),torch.min(pre_image))
    # print('image after preprocessing : ',image.size(),torch.max(image),torch.min(image))
    debug_image = True
    if debug_image:
        img = pre_image.to(torch.uint8)
        im3 = Image.fromarray(img[0].cpu().numpy(), 'RGB').resize((512,512))
        # plt.subplot(2,2,3)
        # plt.imshow(im3)
        im3.save('out/vesseltest/'+str(3)+'.jpg')
    # print('pre image grad: ',pre_image.requires_grad)
    lesion_mask = eval_model(model, pre_image,device)
    # print('lesion image grad: ',lesion_mask.requires_grad)
        
    if debug_image:
        # plt.subplot(2,2,2)
        # plt.imshow((mask>0.5).astype('uint8'))
        mask = lesion_mask.detach().cpu().numpy()
        mask=(mask>0.5).astype('uint8')*255
        # mask_gt = (vessel_gt.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        mask = Image.fromarray(mask).resize((512,512))
        mask.save('out/vesseltest/'+str(2)+'.jpg')
    # print(np.max(mask),np.min(mask),'lesion mask max min')
    ## (0~1)
    # mask = (lesion_mask>0.5)
    # print('lesion mask >: ',mask.requires_grad)
    # mask = mask.type(torch.uint8)
    # tt = lesion_mask.type(torch.uint8)
    # print('lesion mask type: ',tt.requires_grad)
    # print('lesion mask grad: ',mask.requires_grad,lesion_mask.requires_grad)
    return  lesion_mask #(mask>0.5).astype('uint8')

def lesion_loss_init(device):
    from optparse import OptionParser

    import random
    import copy

    import torch.backends.cudnn as cudnn
    import torch.nn as nn
    from torch import optim
    from torch.optim import lr_scheduler

    import config_gan_ex as config
    from hednet import HNNNet
    from dnet import DNet
    # from ..stylespace.ThirdPart.DR-segmentation.HEDNet_cGAN.utils import get_images
    # from myutils import get_images
    # from dataset import IDRIDDataset


    tseed = 1234
    model_pth = './ThirdPart/DR-segmentation/HEDNet_cGAN/results/model_True.pth.tar'
    lesion_type = 'EX'

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(tseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tseed)
    np.random.seed(tseed)
    random.seed(tseed)

    model = HNNNet(pretrained=True, class_number=2)

    resume = model_pth

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume,map_location=device)
        start_epoch = checkpoint['epoch']+1
        start_step = checkpoint['step']
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            model.load_state_dict(checkpoint['g_state_dict'])
        print('Model loaded from {}'.format(resume))
    else:
        print("=> no checkpoint found at '{}'".format(resume))


    for name,param in model.named_parameters():
        param.requires_grad = False
    return model



#####################################################################################################
def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    vesselseg,dicebce_loss = vessel_loss_init(device)
    vesselseg.to(device)
    vesselseg.eval()
    lesionsegmodel = lesion_loss_init(device)
    lesionsegmodel.to(device)
    lesionsegmodel.eval()
    pixelloss = torch.nn.L1Loss()
    
    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # for name,param in vgg16.named_parameters():
    #     print(param.requires_grad)
    ############ all False
    # for name,param in lesionsegmodel.named_parameters():
    #     print(param.requires_grad)
    ############ all False
    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    # ttmp_gt_img = target_images.clone()
    # ttmp_gt_img.requires_grad = True
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)
    # print('target_features grad: ',target_features.requires_grad)
    ##### gt vessel seg ######
    # tmp_gt_img = target_images.detach().clone()
    tmp_gt_img = target_images.clone()
    # print('tmp_gt_img target grad: ',tmp_gt_img.requires_grad,target_images.requires_grad) ##  False False
    maxv = torch.max(tmp_gt_img)
    minv = torch.min(tmp_gt_img)
    absmax = max(maxv,abs(minv))
    tmp_gt_img = tmp_gt_img/absmax  ## RGB->BGR (2,1,0)
    # print(tmp_gt_img.size(),'gt shape')
    tmp_gt_img = tmp_gt_img[:,[2,1,0],:,:]

    gt_vessel_img = tmp_gt_img.clone()
    gt_vessel_img = (gt_vessel_img+1.0)*0.5
    # tmp_gt_img = (tmp_gt_img[:,[2,1,0],:,:]+1.0)*0.5
    # print(torch.max(tmp_gt_img),torch.min(tmp_gt_img),' min max tmp_gt_img...... ')
    # vessel_gt = vesselseg(gt_vessel_img).detach() ## unetseg input should be (0~1)
    
    if vesselseg.training:
        print('in training model')
    else:
        print('in eval model')
    vessel_gt = vesselseg(gt_vessel_img).detach()
    # print('vseelgt,gtvessel grad: ',vessel_gt.requires_grad,gt_vessel_img.requires_grad,target_features.requires_grad)## True False False
    # print(vessel_gt.size())
        
    out = torch.sigmoid(vessel_gt)  ### value 0-1.0
    parsing = out.squeeze().cpu().detach().numpy()
    # print(np.max(parsing),np.min(parsing),'parsing') ### (max:0.99,min:0.01)
    mask = parsing > 0.5       
    mask=mask.astype('uint8')*255
        
    # mask_gt = (vessel_gt.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    mask = Image.fromarray(mask).resize((512,512))
    mask.save('out/vesseltest/'+str(1)+'.jpg')
    
    
    
    # print('tmp_gt_img grad: ',tmp_gt_img.requires_grad)
    #### gt lesion seg  ########
    # lesion_gt_img = tmp_gt_img.detach().clone()
    lesion_gt_img = tmp_gt_img.clone()
    
    lesion_gt = get_lesion_mask(lesion_gt_img,lesionsegmodel,device).detach() ################# ok?
    # print('lesion_gt_img,lesiongt grad: ',lesion_gt_img.requires_grad,lesion_gt.requires_grad)
    tmask = lesion_gt.cpu().numpy()
    tmask = Image.fromarray(((tmask>0.5).astype(np.uint8))*255).resize((512,512))
    tmask.save('out/vesseltest/'+'lesiongt.jpg')
    # lesion_gt = torch.from_numpy(lesion_gt)
    #####################################################################################

   
    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')
        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        # tmp_gen_img = synth_images.detach().clone() ### (-1.0~1.0)
        tmp_gen_img = synth_images.clone() ### (-1.0~1.0)
        # tmp_lesion_gen_img = synth_images.detach().clone()
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
            
        
        # tmp_gt_img = real_img.detach()
        maxv = torch.max(tmp_gen_img)
        minv = torch.min(tmp_gen_img)
        absmax = max(maxv,abs(minv))
        # print(tmp_gen_img.size(),'gen shape')
        tmp_gen_img = tmp_gen_img/absmax  ## RGB->BGR (2,1,0)
        tmp_gen_img = tmp_gen_img[:,[2,1,0],:,:]
        # tmp_gen_img = (tmp_gen_img[:,[2,1,0],:,:]+1.0)*0.5  ### value:0~1.0
        # print(torch.max(tmp_gen_img),torch.min(tmp_gen_img),' min max tmp_gen_img...... ')
        # tmp_gt_img = (tmp_gt_img[:,[2,1,0],:,:]+1.0)*0.5

        gen_vessel_img = tmp_gen_img.clone()
        gen_vessel_img = (gen_vessel_img+1.0)*0.5
        # vessel_gen = vesselseg(gen_vessel_img).detach() ## unetseg input should be (0~1)
        vessel_gen = vesselseg(gen_vessel_img)
        # vessel_gt = self.vesselseg(tmp_gt_img)
        
        # print('vessel_gen_img,vesselgen grad: ',tmp_gen_img.requires_grad,gen_vessel_img.requires_grad,vessel_gen.requires_grad)
        gen_lesion_img = tmp_gen_img.clone()
        lesion_gen = get_lesion_mask(gen_lesion_img,lesionsegmodel,device)
        # lesion_gen = torch.from_numpy(lesion_gen)
        # print('lesion_gen_img,lesiongen grad: ',gen_lesion_img.requires_grad,lesion_gen.requires_grad)
        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()
        # print('vessel grad, vgg grad: ',synth_images.requires_grad,synth_features.requires_grad,target_features.requires_grad,gen_vessel_img.requires_grad,vessel_gen.requires_grad,vessel_gt.requires_grad) 
        ##### True True False True True True
        # for name,param in vgg16.named_parameters():
        #     print('in loop vgg16',param.requires_grad)
        # for name,param in vesselseg.named_parameters():
        #     print('in loop vesselseg',param.requires_grad)
        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        seg_loss = 0.5*dicebce_loss(vessel_gen,vessel_gt)
        lesion_loss = 1.0*dicebce_loss(lesion_gen,lesion_gt)
        pixel_loss = pixelloss(synth_images,target_images) 
        # pixel_loss = F.softmax(pixel_loss_t)
        
        print(lr)
        # print(lesion_loss.requires_grad,seg_loss.requires_grad,dist.requires_grad)
        ## False,True,True->True,True,True
        loss = dist + reg_loss * regularize_noise_weight   + lesion_loss
        # loss = lesion_loss
        # Step
        optimizer.zero_grad(set_to_none=True)
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f} segloss {float(seg_loss):<5.2f} lesionloss {float(lesion_loss):<5.2f} pixel_loss {float(pixel_loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--fps',                    help='Frames per second of final video', default=30, show_default=True)
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    fps: int,
):
    """Project given image to the latent space of pretrained network pickle.
    Examples:
    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda:0')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=fps, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.save(f'{outdir}/projected_w.npy', projected_w.unsqueeze(0).cpu().numpy())

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------