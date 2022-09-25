import argparse
import logging
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset

from preact_resnet import PreActResNet18
from wideresnet import WideResNet
from vgg import VGG16,VGG19
from densenet import DenseNet121
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='/mnt/storage0_8/torch_datasets/cifar-data', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--model', default='pre', type=str, choices=['pre', 'wide', 'vgg16', 'vgg19', 'dense'])
    parser.add_argument('--wide-factor', default=10, type=int, help='Widen factor')
    parser.add_argument('--lr-schedule', default='multistep', type=str, choices=['cyclic', 'flat', 'multistep'])
    parser.add_argument('--lr-min', default=0.0, type=float)
    parser.add_argument('--lr-max', default=0.05, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--gamma', default=1.01, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--normalization', default='std', type=str, choices=['std', '01','+-1'])
    parser.add_argument('--init-tau', default=8, type=float)
    parser.add_argument('--max-iteration', default=7, type=int)
    parser.add_argument('--delta-init', default='uniform', choices=['zero', 'uniform', 'normal', 'bernoulli'],
        help='Perturbation initialization method')
    parser.add_argument('--alpha', default=2., type=float, help='Step size')
    parser.add_argument('--fname', default='output', type=str)
    parser.add_argument('--out-dir', default='mpgd_out', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save-model', action='store_true')
    return parser.parse_args()

def main():
    args = get_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, args.fname+'.log')
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=logfile,
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO)

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
    epsilon = (args.epsilon / 255.)
    alpha = (args.alpha / 255.)
    cur_alpha = (args.alpha / 255.)
    init_tau = (args.init_tau / 255.)
    if args.normalization == 'std':
        mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar10_std).view(3,1,1).cuda()
    elif args.normalization == '01':
        mu = torch.tensor((0.,0.,0.)).view(3,1,1).cuda()
        std = torch.tensor((1.,1.,1.)).view(3,1,1).cuda()
    elif args.normalization == '+-1':
        mu = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).cuda()
        std = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).cuda()
    if args.model == 'pre':
        model = PreActResNet18().cuda()
    elif args.model == 'vgg19':
        model = VGG19().cuda()
    elif args.model == 'vgg16':
        model = VGG16().cuda()
    elif args.model == 'dense':
        model = DenseNet121().cuda()
    elif args.model == 'wide':
        model = WideResNet(34, 10, widen_factor=args.wide_factor, dropRate=0.0)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'flat':
        lr_lamdbda = lambda t: 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lr_lamdbda)
    elif args.lr_schedule == 'multistep':
        if args.model != 'wide':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 4 / 5], gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    nb_attack_iteration = 0
    highest_idx = 0
    highest_acc = 0
    train_time = 0
    logger.info('Epoch \t Seconds \t LR \t BP\t Train Loss \t Train Acc \t Val Acc \t MG \t PGD Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        grad_mag = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            
            delta = torch.zeros_like(X).cuda()
            if args.delta_init == 'uniform' and args.init_tau > 0.0:
                delta.uniform_(-init_tau, init_tau)
            elif args.delta_init == 'normal' and args.init_tau > 0.0:
                delta.normal_(-init_tau, init_tau)
            elif args.delta_init == 'bernoulli' and args.init_tau > 0.0:
                temp = torch.sign(torch.randn_like(X))
                delta.data = init_tau*temp
            delta= clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True

            if nb_attack_iteration != 0:
                for rr in range(nb_attack_iteration):
                    output = model(normalize(X + delta,mu,std))
                    loss = F.cross_entropy(output, y)
                    loss.backward()
                    grad = delta.grad.detach()
                    delta.data = torch.clamp(delta + cur_alpha * torch.sign(grad), min=-epsilon, max=epsilon)
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    delta.grad.zero_()
                    # if rr + 1 < nb_attack_iteration:
                        # continue
                    # grad_mag += torch.sum(grad.abs()*std)

            delta = delta.detach()
            delta.requires_grad = True
            output = model(normalize(X + delta,mu,std))
            loss = F.cross_entropy(output, y)
            opt.zero_grad()
            loss.backward()
            delta_grad = delta.grad.detach()
            grad_mag += torch.sum(delta_grad.abs()*std)
            opt.step()

            scheduler.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - start_epoch_time
        train_time += epoch_time
        lr = scheduler.get_last_lr()[0]


        # Evaluation
        if args.model == 'pre':
            model_test = PreActResNet18().cuda()
        elif args.model == 'vgg19':
            model_test = VGG19().cuda()
        elif args.model == 'vgg16':
            model_test = VGG16().cuda()
        elif args.model == 'dense':
            model_test = DenseNet121().cuda()
        elif args.model == 'wide':
            model_test = WideResNet(34, 10, widen_factor=args.wide_factor, dropRate=0.0)
        model_test = torch.nn.DataParallel(model_test).cuda()
        model_test.load_state_dict(model.state_dict())
        model_test.float()
        model_test.eval()

        val_adv_loss, val_adv_acc = evaluate_pgd(test_loader, model_test, mu, std, 10, 1, val=20, use_CWloss=True)
        val_loss, val_acc = evaluate_standard(test_loader, model_test, mu, std, val=20)
        # logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f\t %.4f \t %.4f',
            # epoch, epoch_time, lr, nb_attack_iteration, train_loss/train_n, train_acc/train_n, val_acc,
            # grad_mag, val_adv_acc)
        logger.info(f'{epoch}\t{epoch_time:.1f}\t{lr:.4f}\t{nb_attack_iteration+1:d}\t{train_loss/train_n:.4f}\t{train_acc/train_n:.4f}\t{val_acc:.4f}\t{grad_mag:.4f}\t{val_adv_acc:.4f}')

        if epoch == 0:
            nb_attack_iteration += 1
            cur_alpha = epsilon / nb_attack_iteration
            if cur_alpha < alpha:
                cur_alpha = alpah
        elif epoch == 1:
            increase_threshold = args.gamma * grad_mag
        # elif ((grad_mag > increase_threshold and nb_attack_iteration < args.max_iteration)
        #         or (args.alpha * nb_attack_iteration < args.epsilon)):

        elif (grad_mag > increase_threshold and nb_attack_iteration < args.max_iteration):
            increase_threshold = args.gamma * grad_mag
            nb_attack_iteration += 1
            cur_alpha = epsilon / nb_attack_iteration
            if cur_alpha < alpha:
                cur_alpha = alpah

        if val_adv_acc > highest_acc and args.save_model:
            highest_acc = val_adv_acc
            highest_idx = epoch
            torch.save(model.state_dict(), os.path.join(args.out_dir, f'model_{args.model}.pth'))
    logger.info('Total train time: %.4f minutes', (train_time)/60)
    logger.info(f'Best checkpoint at {highest_idx}, {highest_acc}')

if __name__ == "__main__":
    main()