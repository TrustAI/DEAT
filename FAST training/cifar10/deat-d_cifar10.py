import argparse
import logging
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import apex.amp as amp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from preact_resnet import PreActResNet18
from wideresnet import WideResNet
from utils import *
from pgds import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'flat'])
    parser.add_argument('--lr-min', default=0.0, type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--delay', default=3, type=int)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--init-tau', default=8, type=float)
    parser.add_argument('--max-iteration', default=5, type=int)
    parser.add_argument('--delta-init', default='uniform', choices=['zero', 'uniform', 'normal', 'bernoulli'],
        help='Perturbation initialization method')
    parser.add_argument('--alpha', default=10., type=float, help='Step size')
    parser.add_argument('--fname', default='output', type=str)
    parser.add_argument('--out-dir', default='deatd_out', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model-struc', default='pre', type=str, choices=['pre', 'wide'])
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively ad just the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
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
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    init_tau = (args.init_tau / 255.) / std

    if args.model_struc == 'pre':
        model = PreActResNet18().cuda()
    elif args.model_struc == 'wide':
        model = WideResNet(34, 10, widen_factor=10, dropRate=0.0).cuda()
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, opt = amp.initialize(model, opt, **amp_args)

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'flat':
        lr_lamdbda = lambda t: 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lr_lamdbda)
        
    # Training
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0

        current_iteration = epoch // args.delay + 1
        nb_replay = current_iteration if current_iteration < args.max_iteration else args.max_iteration

        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            
            delta = torch.zeros_like(X).cuda()
            if args.delta_init == 'uniform' and args.init_tau > 0.0:
                for i in range(len(epsilon)):
                    delta[:, i, :, :].uniform_(-init_tau[i][0][0].item(), init_tau[i][0][0].item())
            elif args.delta_init == 'normal' and args.init_tau > 0.0:
                for i in range(len(epsilon)):
                    delta[:, i, :, :].normal_(0, init_tau[i][0][0].item())
            elif args.delta_init == 'bernoulli' and args.init_tau > 0.0:
                temp = torch.sign(torch.randn_like(X))
                for i in range(len(epsilon)):
                    delta.data[:, i, :, :] = init_tau[i][0][0].item()*temp[:, i, :, :]

            delta.requires_grad = True
            
            for _ in range(nb_replay):
                output = model(X + delta[:X.size(0)])
                loss = F.cross_entropy(output, y)
                opt.zero_grad()
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                grad = delta.grad.detach()
                delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                opt.step()
                delta.grad.zero_()

            scheduler.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
        if args.save_model:
            torch.save(model.state_dict(), os.path.join(args.out_dir, f'model{epoch}.pth'))
    train_time = time.time()
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    if args.model_struc == 'pre':
        model_test = PreActResNet18().cuda()
    elif args.model_struc == 'wide':
        model_test = WideResNet(34, 10, widen_factor=10, dropRate=0.0).cuda()
    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    cw_loss, cw_acc = evaluate_pgd(test_loader, model_test, 20, 1, loss='cw')
    pgd20_loss, pgd20_acc = evaluate_pgd(test_loader, model_test, 20, 1)
    pgd100_loss, pgd100_acc = evaluate_pgd(test_loader, model_test, 100, 1)
    fgsm_loss, fgsm_acc = evaluate_pgd(test_loader, model_test, 1, 1, random_init=False)
    test_loss, test_acc = clean_evaluate(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t FGSM Loss \t FGSM Acc \t PGD20 Loss \t PGD20 Acc \t PGD100 Loss \t PGD100 Acc \t CW20 Loss \t CW20 Acc')
    logger.info('{:.4f} \t\t {:.4f} \t  {:.4f} \t  {:.4f} \t  {:.4f} \t  {:.4f} \t  {:.4f} \t  {:.4f} \t  {:.4f} \t  {:.4f}'.format(
        test_loss, test_acc, fgsm_loss, fgsm_acc, pgd20_loss, pgd20_acc, pgd100_loss, pgd100_acc, cw_loss, cw_acc))

if __name__ == "__main__":
    main()
