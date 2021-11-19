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
from wideresnet import WideResNet
from preact_resnet import PreActResNet18
from utils import (evaluate_pgd,evaluate_standard,clamp,normalize)

CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

mu = torch.tensor(CIFAR100_MEAN).view(3,1,1).cuda()
std = torch.tensor(CIFAR100_STD).view(3,1,1).cuda()

upper_limit = 1.
lower_limit = 0.

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='/mnt/storage0_8/torch_datasets/cifar-data', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr-schedule', default='multistep', type=str, choices=['cyclic', 'flat', 'multistep'])
    parser.add_argument('--lr-min', default=0.0, type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--model', default='wide', type=str, choices=['pre', 'wide'])
    parser.add_argument('--wide-factor', default=10, type=int, help='Widen factor')
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--gamma', default=1.01, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--init-tau', default=8, type=float)
    parser.add_argument('--max-iteration', default=8, type=int)
    parser.add_argument('--delta-init', default='uniform', choices=['zero', 'uniform', 'normal', 'bernoulli'],
        help='Perturbation initialization method')
    parser.add_argument('--alpha', default=10., type=float, help='Step size')
    parser.add_argument('--fname', default='output', type=str)
    parser.add_argument('--out-dir', default='mdeat_out_cifar100', type=str, help='Output directory')
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

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    num_workers = 2
    train_dataset = datasets.CIFAR100(
        args.data_dir, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR100(
        args.data_dir, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    epsilon = (args.epsilon / 255.)
    alpha = (args.alpha / 255.)
    init_tau = (args.init_tau / 255.)


    if args.model == 'pre':
        # model = PreActResNet18().cuda()
        raise('Not support in Cifar-100, pls use WideResNet!')
    elif args.model == 'wide':
        model = WideResNet(34, 100, widen_factor=args.wide_factor).cuda()
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
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 4 / 5], gamma=0.1)
        
    # Training
    nb_replay = 1
    highest_acc = 0
    train_time = 0
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc \t Val Acc \t MG \t PGD Acc')
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

            for rr in range(nb_replay):
                output = model(normalize(X + delta[:X.size(0)]))
                loss = F.cross_entropy(output, y)
                opt.zero_grad()
                loss.backward()
                grad = delta.grad.detach()
                delta.data =  torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                opt.step()
                if rr + 1 < nb_replay:
                    delta.grad.zero_()
                
            grad_mag += torch.sum(grad.abs()*std)
            scheduler.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - start_epoch_time
        train_time += epoch_time
        lr = scheduler.get_last_lr()[0]
        if epoch == 0:
            nb_replay += 1
        elif epoch == 1:
            increase_threshold = args.gamma * grad_mag
        elif grad_mag > increase_threshold and nb_replay < args.max_iteration:
            increase_threshold = args.gamma * grad_mag
            nb_replay += 1
        
        # Evaluation
        if args.model == 'pre':
            # model_test = PreActResNet18().cuda()
            raise('Not support in Cifar-100, pls use WideResNet!')
        elif args.model == 'wide':
            model_test = WideResNet(34, 100, widen_factor=args.wide_factor).cuda()
        model_test.load_state_dict(model.state_dict())
        model_test.float()
        model_test.eval()

        val_adv_loss, val_adv_acc = evaluate_pgd(test_loader, model_test, 10, 1, val=20, use_CWloss=True)
        val_loss, val_acc = evaluate_standard(test_loader, model_test, val=20)
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f\t %.4f \t %.4f',
            epoch, epoch_time, lr, train_loss/train_n, train_acc/train_n, val_acc,
            grad_mag, val_adv_acc)

        if val_adv_acc > highest_acc and args.save_model:
            highest_acc = val_adv_acc
            highest_idx = epoch
            torch.save(model.state_dict(), os.path.join(args.out_dir, f'model_{args.model}.pth'))
    logger.info('Total train time: %.4f minutes', (train_time)/60)
    logger.info(f'Best checkpoint at {highest_idx}, {highest_acc}')

if __name__ == "__main__":
    main()
