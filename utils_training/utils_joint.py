import pdb
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
def unNormMap1D_to_NormMap2D(idx_B_Avec, scale_factor, delta4d=None, k_size=1, do_softmax=False, scale='centered', return_indices=False,
                    invert_matching_direction=False):
    to_cuda = lambda x: x.cuda() if idx_B_Avec.is_cuda else x
    batch_size, sz = idx_B_Avec.size()

    w = sz // scale_factor
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

    JA, IA = Variable(to_cuda(torch.LongTensor(JA).contiguous().view(1, -1))), Variable(to_cuda(torch.LongTensor(IA).contiguous().view(1, -1)))
    # JB, IB = Variable(to_cuda(torch.LongTensor(JB).view(1, -1))), Variable(to_cuda(torch.LongTensor(IB).view(1, -1)))

    iA = IA.contiguous().view(-1)[idx_B_Avec.contiguous().view(-1)].contiguous().view(batch_size, -1)
    jA = JA.contiguous().view(-1)[idx_B_Avec.contiguous().view(-1)].contiguous().view(batch_size, -1)
    # iB = IB.expand_as(iA)
    # jB = JB.expand_as(jA)

    xA=XA[iA.contiguous().view(-1),jA.contiguous().view(-1)].contiguous().view(batch_size,-1)
    yA=YA[iA.contiguous().view(-1),jA.contiguous().view(-1)].contiguous().view(batch_size,-1)
    # xB=XB[iB.view(-1),jB.view(-1)].view(batch_size,-1)
    # yB=YB[iB.view(-1),jB.view(-1)].view(batch_size,-1)

    xA_WTA = xA.contiguous().view(batch_size, 1, h, w)
    yA_WTA = yA.contiguous().view(batch_size, 1, h, w)
    Map2D_WTA = torch.cat((xA_WTA, yA_WTA), 1).float()

    return Map2D_WTA



def warp_from_NormMap2D(x, NormMap2D):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid


    vgrid = NormMap2D.permute(0, 2, 3, 1).contiguous()
    output = nn.functional.grid_sample(x, vgrid, align_corners=True) #N,C,H,W
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)
    #
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    # return output*mask
    return output

def calc_pixelCT_mask(corr_2D, index_2D, mask, temperature):
    B, S, S = index_2D.size()
    nc_BSS = corr_2D.contiguous().view(B*S*S, S*S)
    index_1D = index_2D.view(B*S*S, 1)
    mask_pixelCT = torch.zeros(B*S*S, S*S).bool()
    mask_pixelCT[torch.arange(B*S*S), index_1D.detach().squeeze(1)] = True
    positive = nc_BSS[mask_pixelCT].view(B*S*S, -1)
    negative = nc_BSS[~mask_pixelCT].view(B*S*S, -1)
    print(positive[0], positive[256], positive[512], positive[768])
    # mask1D = torch.zeros(B*S*S, 1).bool()
    mask_label = mask.view(-1, 1).bool()
    # mask1D[mask_label] = True
    mask1D = mask_label.detach().squeeze(1)
    positive = positive[mask1D, :]

    negative = negative[mask1D, :]

    masked_logits = torch.cat([positive, negative], dim=1)


    eps_temp = 1e-6
    masked_logits = (masked_logits / temperature) + eps_temp
    src_num_fgnd = mask.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True).sum(dim=0, keepdim=True)
    src_num_fgnd_label = src_num_fgnd.item()
    GPU_NUM = torch.cuda.current_device()
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    labels = torch.zeros(int(src_num_fgnd_label), device=device, dtype=torch.int64)

    loss_pixelCT = F.cross_entropy(masked_logits, labels, reduction='sum')
    loss_pixelCT = (loss_pixelCT / src_num_fgnd).sum()
    return loss_pixelCT