import os
import pdb
import sys
from operator import add
from functools import reduce, partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from timm.models.layers import DropPath, trunc_normal_
import torchvision.models as models

from models.feature_backbones import resnet, vgg
from models.mod import FeatureL2Norm, unnormalise_and_convert_mapping_to_flow
from torch.autograd import Variable
from utils_training.showPlot import plot_test_map_mask_img
from utils_training.utils_joint import calc_pixelCT_mask
import matplotlib.pyplot as plt
from utils_training.utils_joint import unNormMap1D_to_NormMap2D, warp_from_NormMap2D

r'''
Modified timm library Vision Transformer implementation
https://github.com/rwightman/pytorch-image-models
'''

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiscaleBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_multiscale = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        '''
        Multi-level aggregation
        '''
        B, N, H, W = x.shape
        if N == 1:
            x = x.flatten(0, 1)
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x.view(B, N, H, W)
        x = x.flatten(0, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp2(self.norm4(x)))
        x = x.view(B, N, H, W).transpose(1, 2).flatten(0, 1) 
        x = x + self.drop_path(self.attn_multiscale(self.norm3(x)))
        x = x.view(B, H, N, W).transpose(1, 2).flatten(0, 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, N, H, W)
        return x


class TransformerAggregator(nn.Module):
    def __init__(self, num_hyperpixel, img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        #positional_embedding
        self.pos_embed_x = nn.Parameter(torch.zeros(1, num_hyperpixel, 1, img_size, embed_dim // 2))
        self.pos_embed_y = nn.Parameter(torch.zeros(1, num_hyperpixel, img_size, 1, embed_dim // 2))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            MultiscaleBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.proj = nn.Linear(embed_dim, img_size ** 2)
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed_x, std=.02)
        trunc_normal_(self.pos_embed_y, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, corr, source, target):
        B = corr.shape[0]
        x = corr.clone()
        #corr: (B, hyper_pixel_id, sxs, sxs) source: (B, hyper_num, SxS, C), target: (B, hyper_num, SxS, C)
        #corr: (B, 8, 256, 256), source: (B, 8, 256, 128), target: (B, 8, 256, 128)
        #self.pos_embed_x.shape : torch.Size([1, 8, 1, 16, 192])
        # nn.Parameter(torch.zeros(1, num_hyperpixel, 1, img_size, embed_dim // 2)
        pos_embed = torch.cat((self.pos_embed_x.repeat(1, 1, self.img_size, 1, 1), self.pos_embed_y.repeat(1, 1, 1, self.img_size, 1)), dim=4)
        #pos_embed.shape : torch.Size([1, 8, 16, 16, 192]) =>cat: torch.Size([1, 8, 16, 16, 384])
        pos_embed = pos_embed.flatten(2, 3)
        #pos_embed.shape : torch.Size([1, 8, 256, 384(256+128)])
        x = torch.cat((x.transpose(-1, -2), target), dim=3) + pos_embed
        # x.shape: torch.Size([4, 8, 256, 384])
        x = self.proj(self.blocks(x)).transpose(-1, -2) + corr  # swapping the axis for swapping self-attention.

        x = torch.cat((x, source), dim=3) + pos_embed
        x = self.proj(self.blocks(x)) + corr 


        return x.mean(1)

class adap_layer_feat3(nn.Module):
    def __init__(self):
        super(adap_layer_feat3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        GPU_NUM = torch.cuda.current_device()
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        print("find_correspondence_gpu:", device)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.conv1.cuda()
            self.conv2.cuda()

    def forward(self, feature):
        feature = feature + self.conv1(feature)
        feature = feature + self.conv2(feature)
        return feature

class FeatureExtractionHyperPixel(nn.Module):
    def __init__(self,hyperpixel_ids, feature_size, freeze=True):
        super().__init__()
        self.backbone = resnet.resnet101(pretrained=True)
        self.feature_size = feature_size
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        nbottlenecks = [3, 4, 23, 3]
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.hyperpixel_ids = hyperpixel_ids
        self.adap_layer_feat3 = adap_layer_feat3()
    def forward(self, img):
        r"""Extract desired a list of intermediate features"""

        feats = []

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in self.hyperpixel_ids:
                feats.append(feat.clone())
                #if hid + 1 == max(self.hyperpixel_ids):
                #    break
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat in enumerate(feats):
            feat = self.adap_layer_feat3(feat)
            feats[idx] = F.interpolate(feat, self.feature_size, None, 'bilinear', True)
        return feats

class FeatureCorrelation(torch.nn.Module):
    def __init__(self, shape='3D', normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape = shape
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        if self.shape == '3D':
            b, c, h, w = feature_A.size()
            # reshape features for matrix multiplication
            feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
            feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_B, feature_A)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        elif self.shape == '4D':
            b, c, hA, wA = feature_A.size()
            b, c, hB, wB = feature_B.size()
            # reshape features for matrix multiplication
            feature_A = feature_A.view(b, c, hA * wA).transpose(1, 2)  # size [b,c,h*w]
            feature_B = feature_B.view(b, c, hB * wB)  # size [b,c,h*w]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_A, feature_B)
            # indexed [batch,row_A,col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b, hA, wA, hB, wB).unsqueeze(1)


        return correlation_tensor



class CATs(nn.Module):
    def __init__(self,
    feature_size=16,
    feature_proj_dim=128,
    depth=4,
    num_heads=6,
    mlp_ratio=4,
    hyperpixel_ids=[0,8,20,21,26,28,29,30],
    freeze=True,
    alpha_1=0.01,
    alpha_2=0.5,
    temperature = 0.01,
    feat_position = 'mutual',
     sym=True,
     neg_all=True,
     use_mask=True,
     save_path=None,
     use_detach=True,
     use_adap=True,
     ablation_type = 'all',
     ablation_by = 'all'):
        super().__init__()
        self.feature_size = feature_size
        #self.feature_proj_dim = 128
        self.feature_proj_dim = feature_proj_dim
        # self.decoder_embed_dim = 384
        self.decoder_embed_dim = self.feature_size ** 2 + self.feature_proj_dim
        
        channels = [64] + [256] * 3 + [512] * 4 + [1024] * 23 + [2048] * 3

        self.feature_extraction = FeatureExtractionHyperPixel(hyperpixel_ids, feature_size, freeze)
        self.proj = nn.ModuleList([
            nn.Linear(channels[i], self.feature_proj_dim) for i in hyperpixel_ids
        ])
        self.decoder = TransformerAggregator(
            img_size=self.feature_size, embed_dim=self.decoder_embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_hyperpixel=len(hyperpixel_ids))
            
        self.l2norm = FeatureL2Norm()
        self.corr_jw = FeatureCorrelation(shape='4D', normalization=False)
        self.corr_jw_3d = FeatureCorrelation(shape='3D', normalization=False)
        self.x_normal = np.linspace(-1,1,self.feature_size)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))
        self.y_normal = np.linspace(-1,1,self.feature_size)
        self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False))

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.temperature = temperature
        self.sym = sym
        self.feat_position = feat_position
        self.neg_all = neg_all
        self.use_mask = use_mask
        self.ablation_type = ablation_type
        self.ablation_by = ablation_by
        self.save_path = save_path
        self.count = 0
    def softmax_with_temperature(self, x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def soft_argmax(self, corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        b,_,h,w = corr.size()
        
        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
        x_normal = self.x_normal.expand(b,w)
        x_normal = x_normal.view(b,w,1,1)
        grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        
        grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
        y_normal = self.y_normal.expand(b,h)
        y_normal = y_normal.view(b,h,1,1)
        grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
        return grid_x, grid_y
    
    def mutual_nn_filter(self, correlation_matrix):
        r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
        corr_src_max = torch.max(correlation_matrix, dim=3, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += 1e-30
        corr_trg_max[corr_trg_max == 0] += 1e-30

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)
    
    def corr(self, src, trg):
        return src.flatten(2).transpose(-1, -2) @ trg.flatten(2)

    def calOcc(self, index_S_Tvec, index_T_Svec):
        batch_size, _, _ = index_S_Tvec.size()  # (B, 2, H, W)

        index1D_index_S_Tvec = index_S_Tvec.view(batch_size, -1)
        index1D_index_T_Svec = index_T_Svec.view(batch_size, -1)

        norm_map2D_S_Tvec = self.unNormMap1D_to_NormMap2D(index1D_index_S_Tvec, 16)  # (B,2,S,S)
        norm_map2D_T_Svec = self.unNormMap1D_to_NormMap2D(index1D_index_T_Svec, 16)  # (B,2,S,S)

        # mask-FB_check (WTA)
        unnorm_flow2D_S_Tvec = unnormalise_and_convert_mapping_to_flow(norm_map2D_S_Tvec)
        unnorm_flow2D_T_Svec = unnormalise_and_convert_mapping_to_flow(norm_map2D_T_Svec)
        unnorm_flow2D_S_Tvec_bw = nn.functional.grid_sample(unnorm_flow2D_T_Svec, norm_map2D_S_Tvec.permute(0, 2, 3, 1))
        unnorm_flow2D_T_Svec_bw = nn.functional.grid_sample(unnorm_flow2D_S_Tvec, norm_map2D_T_Svec.permute(0, 2, 3, 1))

        occ_S_Tvec = self.generate_mask(unnorm_flow2D_S_Tvec, unnorm_flow2D_S_Tvec_bw,
                                   self.alpha_1, self.alpha_2)  # compute: feature_map-based
        occ_T_Svec = self.generate_mask(unnorm_flow2D_T_Svec, unnorm_flow2D_T_Svec_bw,
                                   self.alpha_1, self.alpha_2)  # compute: feature_map-based
        occ_S_Tvec = occ_S_Tvec.unsqueeze(1)
        occ_T_Svec = occ_T_Svec.unsqueeze(1)
        return occ_S_Tvec, occ_T_Svec

    def unNormMap1D_to_NormMap2D(self, idx_B_Avec, fs1, delta4d=None, k_size=1, do_softmax=False, scale='centered',
                                 return_indices=False,
                                 invert_matching_direction=False):
        to_cuda = lambda x: x.cuda() if idx_B_Avec.is_cuda else x
        batch_size, sz = idx_B_Avec.shape
        w = sz // fs1
        h = w
        # fs2: width, fs1: height
        if scale == 'centered':
            XA, YA = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
            # XB, YB = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))

        elif scale == 'positive':
            XA, YA = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
            # XB, YB = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))

        JA, IA = np.meshgrid(range(w), range(h))
        # JB, IB = np.meshgrid(range(w), range(h))

        XA, YA = Variable(to_cuda(torch.FloatTensor(XA))), Variable(to_cuda(torch.FloatTensor(YA)))
        # XB, YB = Variable(to_cuda(torch.FloatTensor(XB))), Variable(to_cuda(torch.FloatTensor(YB)))

        JA, IA = Variable(to_cuda(torch.LongTensor(JA).contiguous().view(1, -1))), Variable(
            to_cuda(torch.LongTensor(IA).contiguous().view(1, -1)))
        # JB, IB = Variable(to_cuda(torch.LongTensor(JB).view(1, -1))), Variable(to_cuda(torch.LongTensor(IB).view(1, -1)))

        iA = IA.contiguous().view(-1)[idx_B_Avec.contiguous().view(-1)].contiguous().view(batch_size, -1)
        jA = JA.contiguous().view(-1)[idx_B_Avec.contiguous().view(-1)].contiguous().view(batch_size, -1)
        # iB = IB.expand_as(iA)
        # jB = JB.expand_as(jA)

        xA = XA[iA.contiguous().view(-1), jA.contiguous().view(-1)].contiguous().view(batch_size, -1)
        yA = YA[iA.contiguous().view(-1), jA.contiguous().view(-1)].contiguous().view(batch_size, -1)
        # xB=XB[iB.view(-1),jB.view(-1)].view(batch_size,-1)
        # yB=YB[iB.view(-1),jB.view(-1)].view(batch_size,-1)

        xA_WTA = xA.contiguous().view(batch_size, 1, h, w)
        yA_WTA = yA.contiguous().view(batch_size, 1, h, w)
        Map2D_WTA = torch.cat((xA_WTA, yA_WTA), 1).float()

        return Map2D_WTA

    def generate_mask(self, flow, flow_bw, alpha_1, alpha_2):

        output_sum = flow + flow_bw
        output_sum = torch.sum(torch.pow(output_sum.permute(0, 2, 3, 1), 2), 3)
        output_scale_sum = torch.sum(torch.pow(flow.permute(0,2,3,1),2),3) + torch.sum(torch.pow(flow_bw.permute(0,2,3,1),2),3)
        occ_thresh = alpha_1 * output_scale_sum + alpha_2
        occ_bw = (output_sum > occ_thresh).float()
        mask_bw = 1. - occ_bw

        return mask_bw

    def calc_pixelCT_mask(self, corr_2D, index_2D, mask, temperature, neg_all, use_mask):
        GPU_NUM = torch.cuda.current_device()
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

        B, S, S = index_2D.size()
        nc_BSS = corr_2D.contiguous().view(B * S * S, S * S)
        #check
        # nc_BSS_max_score, nc_BSS_max = torch.max(nc_BSS, dim=1)
        index_1D = index_2D.view(B * S * S, 1)
        mask_pixelCT = torch.zeros(B * S * S, S * S).bool()
        mask_pixelCT[torch.arange(B * S * S), index_1D.detach().squeeze(1)] = True
        positive = nc_BSS[mask_pixelCT].view(B * S * S, -1)
        negative = nc_BSS[~mask_pixelCT].view(B * S * S, -1)
        num_fgnd = B*S*S

        if use_mask:
            mask_label = mask.view(-1, 1).bool()
            mask1D = mask_label.detach().squeeze(1)
            positive = positive[mask1D, :]
            negative = negative[mask1D, :]
            num_fgnd = torch.sum(mask1D == True)

        masked_logits = torch.cat([positive, negative], dim=1)
        labels = torch.zeros(int(num_fgnd), device=device, dtype=torch.int64)

        eps_temp = 1e-6
        masked_logits = (masked_logits / temperature) + eps_temp

        loss_pixelCT = F.cross_entropy(masked_logits, labels, reduction='sum')
        loss_pixelCT = (loss_pixelCT / num_fgnd).sum()
        return loss_pixelCT

    def plot_test_map_mask_img(self, tgt_img, src_img,
                               index_2D_S_Tvec, index_2D_T_Svec,
                               occ_S_Tvec, occ_T_Svec,
                               scale_factor):  # A_Bvec #B_Avec
        mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
        if tgt_img.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        # src= F.interpolate(input = source_img, scale_factor = scale_img, mode = 'bilinear')
        # tgt= F.interpolate(input = target_img, scale_factor = scale_img, mode = 'bilinear')
        tgt_img = tgt_img.mul(std).add(mean)
        src_img = src_img.mul(std).add(mean)

        _, h, w = index_2D_S_Tvec.size()
        index1D_S_Tvec = index_2D_S_Tvec.view(1, -1)
        norm_map2D_S_Tvec = self.unNormMap1D_to_NormMap2D(index1D_S_Tvec, h)

        index1D_T_Svec = index_2D_T_Svec.view(1, -1)
        norm_map2D_T_Svec = self.unNormMap1D_to_NormMap2D(index1D_T_Svec, h)

        norm_map2D_S_Tvec = F.interpolate(input=norm_map2D_S_Tvec, scale_factor=scale_factor, mode='bilinear',
                                          align_corners=True)
        norm_map2D_T_Svec = F.interpolate(input=norm_map2D_T_Svec, scale_factor=scale_factor, mode='bilinear',
                                          align_corners=True)

        masked_warp_S_Tvec = warp_from_NormMap2D(tgt_img, norm_map2D_S_Tvec)  # (B, 2, H, W)

        masked_warp_T_Svec = warp_from_NormMap2D(src_img, norm_map2D_T_Svec)
        if self.use_mask:
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
        axis[0][0].set_title("tgt_img_" + str(self.count))
        axis[0][1].imshow(src_img)
        axis[0][1].set_title("src_img_" + str(self.count))
        axis[1][0].imshow(masked_warp_T_Svec)
        axis[1][0].set_title("warp_S_Tvec_" + str(self.count))
        axis[1][1].imshow(masked_warp_S_Tvec)
        axis[1][1].set_title("warp_T_Svec_" + str(self.count))
        # plt.show()
        if self.use_mask:
            del mask_img_S_Tvec, mask_img_T_Svec
        del tgt_img, src_img, index1D_S_Tvec, index1D_T_Svec, norm_map2D_S_Tvec, norm_map2D_T_Svec, masked_warp_T_Svec, masked_warp_S_Tvec
        torch.cuda.empty_cache()
        return fig



    def forward(self, target, source, mode = 'train'):
        B, _, H, W = target.size()

        src_feats = self.feature_extraction(source)
        tgt_feats = self.feature_extraction(target)

        corrs = []
        src_feats_proj = []
        tgt_feats_proj = []
        #src_feats.size() : (B, hyperpixel_id, )
        #i = 0 src (4, 64, 16, 16) (4, 256, 256)
        #i = 1 src (4, 1024, 16, 16) (4, 256, 256)
        #i = 7 src (4, 1024, 16, 16) (4, 256, 256)
        # so, len(corrs) = 8,



        for i, (src, tgt) in enumerate(zip(src_feats, tgt_feats)):
            #Style:S, Content:T (src_img is warped like target)
            corr_2d_ST = self.corr(self.l2norm(src), self.l2norm(tgt))
            src = src.detach()
            tgt = tgt.detach()
            #first_check
            # corr_3d_S_Tvec = corr_2d_TS.view(corr_2d_TS.size(0),
            #                                      self.feature_size * self.feature_size,
            #                                      self.feature_size, self.feature_size)
            # corr_3d_T_Svec = corr_2d_TS.transpose(-2,-1).view(corr_2d_TS.size(0),
            #                                      self.feature_size * self.feature_size,
            #                                      self.feature_size, self.feature_size)
            # scores_S_Tvec, index_S_Tvec = torch.max(self.l2norm(corr_3d_S_Tvec), dim=1)
            # scores_T_Svec, index_T_Svec = torch.max(self.l2norm(corr_3d_T_Svec), dim=1)

            #Double_check
            # corr_4d_S_Tvec_jw = self.corr_jw(self.l2norm(src), self.l2norm(tgt))
            # B, ch, f1, f2, f3, f4 = corr_4d_S_Tvec_jw.size()
            # corr_3d_T_Svec_jw = corr_4d_S_Tvec_jw.view(corr_4d_S_Tvec_jw.size(0),
            #                                      self.feature_size * self.feature_size,
            #                                      self.feature_size, self.feature_size)
            # corr_3d_S_Tvec_jw = corr_4d_S_Tvec_jw.view(corr_4d_S_Tvec_jw.size(0),
            #                                     self.feature_size, self.feature_size,
            #                                     self.feature_size * self.feature_size).permute(0, 3, 1, 2)
            # scores_WTA_B, index_WTA_B = torch.max(corr_3d_S_Tvec_jw, dim=1)
            # print(index_WTA_B[0])
            #1D
            # scores_S_Tvec, index_S_Tvec = torch.max(self.l2norm(corr_2d_ST), dim=1)
            # scores_T_Svec, index_T_Svec = torch.max(self.l2norm(corr_2d_ST.transpose(-2,-1)), dim=1)
            # #2D
            # index_S_Tvec = index_S_Tvec.view(index_S_Tvec.size(0), self.feature_size, self.feature_size)
            # index_T_Svec = index_T_Svec.view(index_T_Svec.size(0), self.feature_size, self.feature_size)
            #
            # print(scores_S_Tvec[0,0], scores_S_Tvec[1,0], scores_S_Tvec[2,0], scores_S_Tvec[3,0])
            # occ_S_Tvec, occ_T_Svec = self.calOcc(index_S_Tvec, index_T_Svec)
            # plot_test_map_mask_img(target[0].unsqueeze(0), source[0].unsqueeze(0),
            #                        index_S_Tvec[0].unsqueeze(0), index_T_Svec[0].unsqueeze(0),
            #                        occ_S_Tvec[0].unsqueeze(0), occ_T_Svec[0].unsqueeze(0),
            #                        16, 'feat_masked_{}'.format(self.count))

            corrs.append(corr_2d_ST)
            #src (B, C, S, S)                     (B, C, SxS)                                   (B, SxS, C)
            #src (4, 1024, 16, 16) src.flatten(2) (4, 1024, 256) src.flatten(2).transpose(-1, -2) (4, 256, 1024)
            #self.proj[i] (B, SxS, C)
            #self.proj[i] (4, 256, 128)
            src_feats_proj.append(self.proj[i](src.flatten(2).transpose(-1, -2)))
            tgt_feats_proj.append(self.proj[i](tgt.flatten(2).transpose(-1, -2)))
        # torch.stack : tensor_list -> tensor
        #(4, 8, 256, 128) => (B, hyper_num, SxS, C)
        src_feats = torch.stack(src_feats_proj, dim=1)
        tgt_feats = torch.stack(tgt_feats_proj, dim=1)
        #(4, 8, 256, 256) => (B, hyper_num, SxS, SxS)
        corr = torch.stack(corrs, dim=1)

        corr = self.mutual_nn_filter(corr)

        #Feat_corr#
        if self.feat_position == 'mutual':
            corr_feat = corr.squeeze(1)
        elif self.feat_position == 'feat':
            corr_feat = corr_2d_ST
        #1D
        #(B,S*S, S*S)
        scores_T_Svec, index_T_Svec = torch.max(self.l2norm(corr_feat), dim=1)
        scores_S_Tvec, index_S_Tvec = torch.max(self.l2norm(corr_feat.transpose(-2, -1)), dim=1)
        # print(index_S_Tvec[0])
        # 2D
        index_S_Tvec = index_S_Tvec.view(index_S_Tvec.size(0), self.feature_size, self.feature_size)
        index_T_Svec = index_T_Svec.view(index_T_Svec.size(0), self.feature_size, self.feature_size)
        #occ
        occ_S_Tvec, occ_T_Svec = self.calOcc(index_S_Tvec, index_T_Svec)

        #corr: (B, hyper_pixel_id, sxs, sxs) src_feats: (B, hyper_num, SxS, C), tgt_feats: (B, hyper_num, SxS, C)
        refined_corr = self.decoder(corr.detach(), src_feats, tgt_feats)

        #Agg_corr#
        # 1D
        refined_scores_T_Svec, refined_index_T_Svec = torch.max(self.l2norm(refined_corr), dim=1)
        refined_scores_S_Tvec, refined_index_S_Tvec = torch.max(self.l2norm(refined_corr.transpose(-2, -1)), dim=1)
        # 2D
        refined_index_S_Tvec = refined_index_S_Tvec.view(refined_index_S_Tvec.size(0), self.feature_size, self.feature_size)
        refined_index_T_Svec = refined_index_T_Svec.view(refined_index_T_Svec.size(0), self.feature_size, self.feature_size)

        refined_occ_S_Tvec, refined_occ_T_Svec = self.calOcc(refined_index_S_Tvec, refined_index_T_Svec)
        #vis
        self.count+=1
        # if self.count % 20 == 0:
        #     fig=self.plot_test_map_mask_img(target[0].unsqueeze(0), source[0].unsqueeze(0),
        #                            index_S_Tvec[0].unsqueeze(0), index_T_Svec[0].unsqueeze(0),
        #                            occ_S_Tvec[0].unsqueeze(0), occ_T_Svec[0].unsqueeze(0),
        #                            scale_factor = 16
        #                          )
        #     save_path = os.path.join(self.save_path, 'img')
        #     plot_name = 'feat_masked_{}'.format(self.count)
        #     if not os.path.isdir(save_path):
        #         os.mkdir(save_path)
        #     fig.savefig('{}/{}.png'.format(save_path, plot_name),
        #                 bbox_inches='tight')
        #
        #     fig = self.plot_test_map_mask_img(target[0].unsqueeze(0), source[0].unsqueeze(0),
        #                            refined_index_S_Tvec[0].unsqueeze(0), refined_index_T_Svec[0].unsqueeze(0),
        #                            refined_occ_S_Tvec[0].unsqueeze(0), refined_occ_T_Svec[0].unsqueeze(0),
        #                            scale_factor=16
        #                            )
        #     plot_name = 'agg_masked_{}'.format(self.count)
        #     fig.savefig('{}/{}.png'.format(save_path, plot_name),
        #                 bbox_inches='tight')
        #     del fig
        GPU_NUM = torch.cuda.current_device()
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        loss_S_Tvec_feat_by_feat = torch.tensor([0.0], requires_grad=True, device= device)
        loss_S_Tvec_feat_by_agg = torch.tensor([0.0], requires_grad=True, device= device)
        loss_S_Tvec_agg_by_feat = torch.tensor([0.0], requires_grad=True, device= device)
        loss_S_Tvec_agg_by_agg = torch.tensor([0.0], requires_grad=True, device= device)
        if self.ablation_type =='feat' or self.ablation_type == 'all':
            loss_S_Tvec_feat_by_feat = self.calc_pixelCT_mask(corr_feat, index_S_Tvec, occ_S_Tvec, self.temperature, self.neg_all, self.use_mask)
            #symmetric
            # loss_T_Svec_feat_by_feat = self.calc_pixelCT_mask(corr_feat.transpose(-2, -1), index_T_Svec, occ_T_Svec, self.temperature, self.neg_all, self.use_mask)
            loss_S_Tvec_feat_by_agg = self.calc_pixelCT_mask(corr_feat, refined_index_S_Tvec, refined_occ_S_Tvec, self.temperature, self.neg_all, self.use_mask)
        if self.ablation_type == 'agg' or self.ablation_type == 'all':
            loss_S_Tvec_agg_by_feat = self.calc_pixelCT_mask(refined_corr, index_S_Tvec, occ_S_Tvec, self.temperature, self.neg_all, self.use_mask)
            # loss_S_Tvec_agg_by_agg = self.calc_pixelCT_mask(refined_corr, refined_index_S_Tvec, refined_occ_S_Tvec, self.temperature, self.neg_all, self.use_mask)

        if mode == 'train':
            return [loss_S_Tvec_feat_by_feat, loss_S_Tvec_feat_by_agg, loss_S_Tvec_agg_by_feat, loss_S_Tvec_agg_by_agg]
        elif mode == 'eval':
            grid_x, grid_y = self.soft_argmax(
                corr_feat.view(B, -1, self.feature_size, self.feature_size))
            refined_grid_x, refined_grid_y = self.soft_argmax(refined_corr.view(B, -1, self.feature_size, self.feature_size))

            flow = torch.cat((grid_x, grid_y), dim=1)
            flow = unnormalise_and_convert_mapping_to_flow(flow)

            refined_flow = torch.cat((refined_grid_x, refined_grid_y), dim=1)
            refined_flow = unnormalise_and_convert_mapping_to_flow(refined_flow)
            return [flow, refined_flow]


