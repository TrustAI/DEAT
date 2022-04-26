from __future__ import print_function
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import logging
from trades import trades_loss
from preact_resnet import PreActResNet18
from wideresnet import WideResNet
from vgg import VGG16,VGG19
from mobilenet import mobilenetV3_small
from utils import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--model', default='pre', type=str, choices=['pre', 'wide', 'vgg16', 'vgg19', 'mobile'])
parser.add_argument('--wide-factor', default=10, type=int, help='Widen factor')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--normalization', default='std', type=str, choices=['std', '01','+-1'])
parser.add_argument('--fname', default='output', type=str)
parser.add_argument('--data-dir', default='/mnt/storage0_8/torch_datasets/cifar-data', type=str)
parser.add_argument('--out-dir', default='trades_out', type=str, help='Output directory')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--save-model', action='store_true')
args = parser.parse_args()

# settings
epsilon = (args.epsilon / 255.)
step_size = (args.step_size / 255.)
if args.normalization == 'std':
    mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
    std = torch.tensor(cifar10_std).view(3,1,1).cuda()
elif args.normalization == '01':
    mu = torch.tensor((0.,0.,0.)).view(3,1,1).cuda()
    std = torch.tensor((1.,1.,1.)).view(3,1,1).cuda()
elif args.normalization == '+-1':
    mu = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).cuda()
    std = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).cuda()
torch.manual_seed(args.seed)
device = torch.device("cuda")
train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    train_n = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=step_size,
                           epsilon=epsilon,
                           mu=mu,std=std,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * target.size(0)
        train_n += target.size(0)
    return train_loss, train_n


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch+1 >= 25:
        lr = args.lr * 0.1
    if epoch+1 >= 40:
        lr = args.lr * 0.01
    # if epoch >= 100:
    #     lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
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
    if args.model == 'pre':
        model = PreActResNet18().cuda()
    elif args.model == 'vgg19':
        model = VGG19().cuda()
    elif args.model == 'vgg16':
        model = VGG16().cuda()
    elif args.model == 'mobile':
        model = mobilenetV3_small().cuda()
    elif args.model == 'wide':
        model = WideResNet(34, 10, widen_factor=args.wide_factor, dropRate=0.0)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    train_time = 0
    highest_acc = 0
    highest_idx = 0
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Val Acc \t PGD Acc')
    for epoch in range(1, args.epochs + 1):
        start_epoch_time = time.time()
        # adjust learning rate for SGD
        cur_lr = adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train_loss, train_n = train(args, model, device, train_loader, optimizer, epoch)
        end_epoch_time = time.time()
        epoch_time = end_epoch_time - start_epoch_time
        train_time += epoch_time

        # Evaluation
        if args.model == 'pre':
            model_test = PreActResNet18().cuda()
        elif args.model == 'vgg19':
            model_test = VGG19().cuda()
        elif args.model == 'vgg16':
            model_test = VGG16().cuda()
        elif args.model == 'mobile':
            model_test = mobilenetV3_small().cuda()
        elif args.model == 'wide':
            model_test = WideResNet(34, 10, widen_factor=args.wide_factor, dropRate=0.0)
        model_test = torch.nn.DataParallel(model_test).cuda()
        model_test.load_state_dict(model.state_dict())
        model_test.float()
        model_test.eval()

        val_adv_loss, val_adv_acc = evaluate_pgd(test_loader, model_test, mu, std, 10, 1, val=20, use_CWloss=True)
        val_loss, val_acc = evaluate_standard(test_loader, model_test, mu, std, val=20)
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time, cur_lr, train_loss/train_n, val_acc, val_adv_acc)

        if val_adv_acc > highest_acc and args.save_model:
            highest_acc = val_adv_acc
            highest_idx = epoch
            torch.save(model.state_dict(), os.path.join(args.out_dir, f'model_{args.model}.pth'))
    logger.info('Total train time: %.4f minutes', (train_time)/60)
    logger.info(f'Best checkpoint at {highest_idx}, {highest_acc}')
if __name__ == '__main__':
    main()
