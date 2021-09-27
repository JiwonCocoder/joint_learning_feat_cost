import pdb
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils_training.utils import flow2kps
from utils_training.evaluation import Evaluator
import matplotlib.pyplot as plt
from torch.autograd import Variable
from matplotlib.patches import Circle


r'''
    loss function implementation from GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''

def writer_grad_flow(named_parameters, writer, writer_position):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                continue
                print(n, "p.grad is None")
            writer.add_scalar('gradient_flow/{}'.format(n), p.grad.abs().mean().data.cpu().numpy(), writer_position)
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                continue
                print(n)

            print(n)
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            print(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads ) +1, linewidth=1, color="k" )
    plt.xticks(range(0 ,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    # save_path = './gradinet_flow'
    # if not os.path.isdir(save_path):
    #     os.mkdir(save_path)
    # fig.savefig('{}/{}.png'.format(save_path, plot_name),
    #             bbox_inches='tight')


def EPE(input_flow, target_flow, sparse=True, mean=True, sum=False):

    EPE_map = torch.norm(target_flow-input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/torch.sum(~mask)


def train_epoch(net,
                optimizer,
                train_loader,
                device,
                epoch,
                train_writer):
    n_iter = epoch*len(train_loader)
    
    net.train()
    running_total_loss = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, mini_batch in pbar:
        optimizer.zero_grad()
        loss_list = net(mini_batch['trg_img'].to(device), mini_batch['src_img'].to(device), mode= 'train')
        Loss = sum(loss_list)
        Loss.backward()
        if i % 10 == 0:
            # plot_grad_flow(net.named_parameters())
            # writer_grad_flow(net.named_parameters(), train_writer, len(mini_batch)*epoch + i)
            train_writer.add_scalar('Loss/loss_S_Tvec_feat_by_feat', loss_list[0], len(mini_batch)*epoch + i)
            train_writer.add_scalar('Loss/loss_S_Tvec_feat_by_agg', loss_list[1], len(mini_batch)*epoch + i)
            train_writer.add_scalar('Loss/loss_S_Tvec_agg_by_feat', loss_list[2], len(mini_batch)*epoch + i)
            train_writer.add_scalar('Loss/loss_S_Tvec_agg_by_agg', loss_list[3], len(mini_batch)*epoch + i)
        del loss_list[:]
        del loss_list
        # optimizer.step()
        running_total_loss += Loss.item()
        train_writer.add_scalar('train_loss_per_iter', Loss.item(), n_iter)
        n_iter += 1
        pbar.set_description(
                'training: R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), Loss.item()))
    running_total_loss /= len(train_loader)

    del mini_batch
    torch.cuda.empty_cache()
    return running_total_loss


def validate_epoch(net,
                   val_loader,
                   device,
                   epoch):
    net.eval()
    running_total_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        pck_array = []
        refined_pck_array = []
        for i, mini_batch in pbar:
            flow_gt = mini_batch['flow'].to(device)
            pred_flow_list = net(mini_batch['trg_img'].to(device),
                                 mini_batch['src_img'].to(device), mode='eval')

            estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow_list[0],
                                     mini_batch['n_pts'].to(device))
            refined_estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow_list[1],
                                             mini_batch['n_pts'].to(device))

            eval_result = Evaluator.eval_kps_transfer(estimated_kps.cpu(), mini_batch)
            refined_eval_result = Evaluator.eval_kps_transfer(refined_estimated_kps.cpu(), mini_batch)

            Loss = EPE(pred_flow_list[0], flow_gt)
            refined_Loss = EPE(pred_flow_list[1], flow_gt)

            pck_array += eval_result['pck']
            refined_pck_array += refined_eval_result['pck']

            running_total_loss += refined_Loss.item()
            pbar.set_description(
                ' refined_validation R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), refined_Loss.item()))
        mean_pck = sum(pck_array) / len(pck_array)
        refined_mean_pck = sum(refined_pck_array) / len(refined_pck_array)

    return running_total_loss / len(val_loader), [mean_pck, refined_mean_pck]

