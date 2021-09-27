'''
    modified training script of GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''

import argparse
import os
import pdb
import pickle
import random
import time
from os import path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from termcolor import colored
from torch.utils.data import DataLoader

from models.cats import CATs
from models.cats_vgg import CATs as CATs_vgg
import utils_training.optimize as optimize
from utils_training.evaluation import Evaluator
from utils_training.utils import parse_list, load_checkpoint, save_checkpoint, boolean_string
from data import download

import admin.settings as ws_settings
from torchsummary import summary

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='CATs Training Script')
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--pretrained', dest='pretrained', default=None,
                       help='path to pre-trained model')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='start epoch')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=32,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--seed', type=int, default=2,
                        help='Pseudo-RNG seed')
                        
    parser.add_argument('--datapath', type=str, default='../Datasets_CATs')
    parser.add_argument('--benchmark', type=str, default='pfpascal', choices=['pfpascal', 'spair'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',
                        help='learning rate (default: 3e-5)')
    parser.add_argument('--lr-backbone', type=float, default=3e-6, metavar='LR',
                        help='learning rate (default: 3e-6)')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine'])
    parser.add_argument('--step', type=str, default='[70, 80, 90]')
    parser.add_argument('--step_gamma', type=float, default=0.5)

    parser.add_argument('--feature-size', type=int, default=16)
    parser.add_argument('--feature-proj-dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--num-heads', type=int, default=6)
    parser.add_argument('--mlp-ratio', type=int, default=4)
    parser.add_argument('--hyperpixel', type=str, default='[28]')
    parser.add_argument('--freeze', type=boolean_string, nargs='?', const=True, default=True)
    parser.add_argument('--augmentation', type=boolean_string, nargs='?', const=True, default=True)
    #mask_hyper_parameters
    parser.add_argument('--alpha_1', type=float, default=0.1)
    parser.add_argument('--alpha_2', type=float, default=5)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--feat_position', type=str, default='mutual', choices=['feat', 'mutual'])
    parser.add_argument('--sym', type=boolean_string, nargs='?', const=True, default=True)
    parser.add_argument('--neg_all', type=boolean_string, nargs='?', const=True, default=True)
    parser.add_argument('--use_mask', type=boolean_string, nargs='?', const=True, default=True)
    parser.add_argument('--use_detach', type=boolean_string, nargs='?', const=True, default=True)
    parser.add_argument('--use_adap', type=boolean_string, nargs='?', const=True, default=True)
    parser.add_argument('--choose_network', type=str, default='resnet', choices=['vgg', 'resnet'])
    parser.add_argument('--ablation_type', type=str, default='agg', choices=['feat', 'agg', 'all'])
    parser.add_argument('--ablation_by', type=str, default='all', choices=['by_feat', 'by_agg', 'all'])

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')

    # Seed
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print("use_cuda:", use_cuda)
    print(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print('Available devices', torch.cuda.device_count())
    print('Current cuda device', torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))
    torch.cuda.set_device(device)
    print('Changed cuda device', torch.cuda.current_device())

    # Initialize Evaluator
    Evaluator.initialize(args.benchmark, args.alpha)
    
    # Dataloader
    download.download_dataset(args.datapath, args.benchmark)
    train_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'trn', args.augmentation, args.feature_size)
    val_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'val', args.augmentation, args.feature_size)
    train_dataloader = DataLoader(train_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_threads,
        shuffle=True)
    val_dataloader = DataLoader(val_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_threads,
        shuffle=False)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # create summary writer
    cur_snapshot = args.name_exp
    save_path=osp.join(args.snapshots, cur_snapshot)
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))

    # Model
    if args.freeze:
        print('Backbone frozen!')
    if args.choose_network == 'resnet':
        model = CATs(
            feature_size=args.feature_size, feature_proj_dim=args.feature_proj_dim,
            depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
            hyperpixel_ids=parse_list(args.hyperpixel), freeze=args.freeze,
            alpha_1 = args.alpha_1, alpha_2 = args.alpha_2, temperature=args.temperature,
            feat_position=args.feat_position, sym=args.sym, neg_all=args.neg_all,
            use_mask=args.use_mask, save_path=save_path,
            use_detach=args.use_detach, use_adap=args.use_adap,
            ablation_type = args.ablation_type, ablation_by = args.ablation_by)
    elif args.choose_network == 'vgg':
        model = CATs_vgg(
            feature_size=args.feature_size, feature_proj_dim=args.feature_proj_dim,
            depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
            hyperpixel_ids=parse_list(args.hyperpixel), freeze=args.freeze,
            alpha_1 = args.alpha_1, alpha_2 = args.alpha_2, temperature=args.temperature,
            feat_position=args.feat_position, sym=args.sym, neg_all=args.neg_all,
            use_mask=args.use_mask, save_path=save_path,
            use_detach=args.use_detach, use_adap=args.use_adap)
    param_model = [param for name, param in model.named_parameters() if 'feature_extraction' not in name]
    param_backbone = [param for name, param in model.named_parameters() if 'feature_extraction' in name]
    #check param_backbone
    param_backbone_name = [name for name, param in model.feature_extraction.backbone.named_parameters()]
    print(param_backbone_name)

    # Optimizer
    if args.ablation_type == 'agg':
        optimizer = optim.AdamW([{'params': param_model, 'lr': args.lr}],
                    weight_decay=args.weight_decay)
    elif args.ablation_type == 'all':
        optimizer = optim.AdamW([{'params': param_model, 'lr': args.lr}, {'params': param_backbone, 'lr': args.lr_backbone}],
                    weight_decay=args.weight_decay)

    # Scheduler
    scheduler = \
        lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6, verbose=True) \
            if args.scheduler == 'cosine' else \
            lr_scheduler.MultiStepLR(optimizer, milestones=parse_list(args.step), gamma=args.step_gamma, verbose=True)
    model = model.to(device)
    # print(model.state_dict()['feature_extraction.backbone.level_4.0.weight'][0])
    # pdb.set_trace()
    if args.pretrained:
        print("here")
        # reload from pre_trained_model
        model_state_dict = load_checkpoint(model, optimizer, scheduler, filename=args.pretrained)
        #now individually transfer the model parts
        print(model_state_dict['level_4.0.weight'][0])
        log = model.feature_extraction.backbone.load_state_dict(model_state_dict, strict=True)
        print(model.state_dict()['feature_extraction.backbone.level_4.0.weight'][0])
        # now individually transfer the optimizer parts...

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        if not os.path.isdir(args.snapshots):
            os.mkdir(args.snapshots)

        if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
            os.makedirs(osp.join(args.snapshots, cur_snapshot))

        with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
            pickle.dump(args, f)

    best_val = 0
    start_epoch = 0

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    # summary(model, input_size=[(3, 256,256), (3, 256, 256)])
    train_started = time.time()

    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        train_loss = optimize.train_epoch(model,
                                 optimizer,
                                 train_dataloader,
                                 device,
                                 epoch,
                                 train_writer)
        train_writer.add_scalar('train loss', train_loss, epoch)
        # train_writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        train_writer.add_scalar('learning_rate_backbone', scheduler.get_lr()[0], epoch)
        print(colored('==> ', 'green') + 'Train average loss:', train_loss)

        val_loss_grid, val_mean_pck = optimize.validate_epoch(model,
                                                       val_dataloader,
                                                       device,
                                                       epoch=epoch)
        print(colored('==> ', 'blue') + 'Val average grid loss :',
              val_loss_grid)
        print('mean PCK is {}'.format(val_mean_pck))
        print(colored('==> ', 'blue') + 'epoch :', epoch + 1)
        test_writer.add_scalar('mean PCK', val_mean_pck[0], epoch)
        test_writer.add_scalar('refined_mean PCK', val_mean_pck[1], epoch)
        test_writer.add_scalar('val loss', val_loss_grid, epoch)
        is_best = val_mean_pck[1] > best_val
        best_val = max(val_mean_pck[1], best_val)
        #save_result
        with open(os.path.join(save_path,'results.txt'),'a+') as file:
            file.write(f'{val_mean_pck[0], val_mean_pck[1]}\n')
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'best_loss': best_val},
                        is_best, save_path, 'epoch_{}.pth'.format(epoch + 1))

    print(args.seed, 'Training took:', time.time()-train_started, 'seconds')
