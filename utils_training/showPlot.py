import datetime
import pdb

import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .utils_joint import unNormMap1D_to_NormMap2D, warp_from_NormMap2D
import io
import cv2
from torchvision.utils import save_image

def plot_test_map_mask_img(tgt_img, src_img,
                           index_2D_S_Tvec, index_2D_T_Svec,
                           occ_S_Tvec, occ_T_Svec,
                           scale_factor, plot_name,
                           save_path, use_mask): #A_Bvec #B_Avec
    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    if tgt_img.is_cuda:
        mean=mean.cuda()
        std=std.cuda()

    # src= F.interpolate(input = source_img, scale_factor = scale_img, mode = 'bilinear')
    # tgt= F.interpolate(input = target_img, scale_factor = scale_img, mode = 'bilinear')
    tgt_img=tgt_img.mul(std).add(mean)
    src_img=src_img.mul(std).add(mean)

    _, h, w = index_2D_S_Tvec.size()
    index1D_S_Tvec = index_2D_S_Tvec.view(1, -1)
    norm_map2D_S_Tvec = unNormMap1D_to_NormMap2D(index1D_S_Tvec, h)

    index1D_T_Svec = index_2D_T_Svec.view(1, -1)
    norm_map2D_T_Svec = unNormMap1D_to_NormMap2D(index1D_T_Svec, h)

    norm_map2D_S_Tvec = F.interpolate(input=norm_map2D_S_Tvec, scale_factor=scale_factor, mode='bilinear', align_corners= True)
    norm_map2D_T_Svec = F.interpolate(input=norm_map2D_T_Svec, scale_factor=scale_factor, mode='bilinear', align_corners= True)

    masked_warp_T_Svec = warp_from_NormMap2D(tgt_img, norm_map2D_T_Svec) #(B, 2, H, W)

    masked_warp_S_Tvec = warp_from_NormMap2D(src_img, norm_map2D_S_Tvec)
    if use_mask:
        mask_img_S_Tvec = F.interpolate(input=occ_S_Tvec.type(torch.float),
                                        scale_factor=scale_factor,
                                        mode='bilinear',
                                        align_corners=True)
        mask_img_T_Svec = F.interpolate(input=occ_T_Svec.type(torch.float),
                                        scale_factor=scale_factor,
                                        mode='bilinear',
                                        align_corners=True)
        masked_warp_T_Svec = mask_img_T_Svec * masked_warp_T_Svec
        masked_warp_S_Tvec = mask_img_S_Tvec * masked_warp_S_Tvec

    tgt_img = tgt_img * 255.0
    src_img = src_img * 255.0
    tgt_img = tgt_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)
    src_img = src_img.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy().astype(np.uint8)

    masked_warp_T_Svec = masked_warp_T_Svec.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    masked_warp_S_Tvec = masked_warp_S_Tvec.data.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    fig, axis = plt.subplots(2, 2, figsize=(50, 50))
    axis[0][0].imshow(tgt_img)
    axis[0][0].set_title("tgt_img_"+ str(plot_name))
    axis[0][1].imshow(src_img)
    axis[0][1].set_title("src_img_" + str(plot_name))
    axis[1][0].imshow(masked_warp_S_Tvec)
    axis[1][0].set_title("warp_S_Tvec_"+ str(plot_name))
    axis[1][1].imshow(masked_warp_T_Svec)
    axis[1][1].set_title("warp_T_Svec_"+ str(plot_name))
    # plt.show()
    # save_path = './WTA_from_Map_Neighbor'
    save_path = os.path.join(save_path, 'img')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    fig.savefig('{}/{}.png'.format(save_path, plot_name),
                bbox_inches='tight')
    if use_mask:
        mask_img_S_Tvec, mask_img_T_Svec

    del tgt_img, src_img, index1D_S_Tvec, index1D_T_Svec, norm_map2D_S_Tvec, norm_map2D_T_Svec, masked_warp_T_Svec, masked_warp_S_Tvec
    torch.cuda.empty_cache()
