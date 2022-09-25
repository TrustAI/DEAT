from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import time
import logging
from preact_resnet import PreActResNet18
from wideresnet import WideResNet
from vgg import VGG16,VGG19
from densenet import DenseNet121
from utils import *
from mart import mmart_loss



parser = argparse.ArgumentParser(description='PyTorch CIFAR MART Defense')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--model', default='pre', type=str, choices=['pre', 'wide', 'vgg16', 'vgg19', 'dense'])
parser.add_argument('--wide-factor', default=10, type=int, help='Widen factor')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr-max', type=float, default=0.05, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=8., 
                    help='perturbation bound')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--alpha', type=float, default=2., 
                    help='step size')
parser.add_argument('--beta', default=6.0,
                    help='weight before kl (misclassified examples)')
parser.add_argument('--gamma', type=float, default=1.01, metavar='G',
                    help='gamma')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--normalization', default='std', type=str, choices=['std', '01','+-1'])
parser.add_argument('--fname', default='output', type=str)
parser.add_argument('--data-dir', default='/mnt/storage0_8/torch_datasets/cifar-data', type=str)
parser.add_argument('--out-dir', default='mmart_out', type=str, help='Output directory')
parser.add_argument('--save-model', action='store_true')
args = parser.parse_args()

# settings
# training settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
epsilon = (args.epsilon / 255.)
alpha = (args.alpha / 255.)
step_size = (args.alpha / 255.)

if args.normalization == 'std':
    mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
    std = torch.tensor(cifar10_std).view(3,1,1).cuda()
elif args.normalization == '01':
    mu = torch.tensor((0.,0.,0.)).view(3,1,1).cuda()
    std = torch.tensor((1.,1.,1.)).view(3,1,1).cuda()
elif args.normalization == '+-1':
    mu = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).cuda()
    std = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).cuda()
train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

def mmart_delta(model,
              x_natural,
              y,
              optimizer,
              step_size,
              epsilon,
              mu,std,
              perturb_steps=10,
              distance='l_inf'):

    if distance == 'l_inf':
        delta = 0.001*torch.randn_like(x_natural).cuda()
        delta = clamp(delta, lower_limit - x_natural, upper_limit - x_natural)
        for rr in range(perturb_steps):
            delta.requires_grad_()

            loss_ce = F.cross_entropy(model(normalize(x_natural+delta,mu,std)), y)
            loss_ce.backward()
            grad = delta.grad.detach()
            
            delta.data =  torch.clamp(delta + step_size * torch.sign(grad), min=-epsilon, max=epsilon)
            delta.data[:x_natural.size(0)] = clamp(delta[:x_natural.size(0)], lower_limit - x_natural, upper_limit - x_natural)
            delta.grad.zero_()
            # if rr + 1 < perturb_steps:
            #     continue
            # grad_mag += torch.sum(grad.abs()*std)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    return delta.detach()


def train(args, model, device, train_loader, num_steps, optimizer, epoch):
    model.train()
    train_acc = 0
    train_loss = 0
    train_n = 0
    beta=args.beta       
    grad_mag = 0
    kl = nn.KLDivLoss(reduction='none')

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # calculate robust loss
        delta = mmart_delta(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=step_size,
                           epsilon=epsilon,
                           mu=mu,std=std,
                           perturb_steps=num_steps,
                           )
        batch_size = len(data)
        delta.requires_grad = True
        x_adv = data.detach() + delta
        # zero gradient
        optimizer.zero_grad()

        logits = model(normalize(data,mu,std))

        logits_adv = model(normalize(x_adv,mu,std))
        train_acc += (logits_adv.max(1)[1] == target).sum().item()

        adv_probs = F.softmax(logits_adv, dim=1)
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

        new_y = torch.where(tmp1[:, -1] == target, tmp1[:, -2], tmp1[:, -1])

        loss_adv = F.cross_entropy(logits_adv, target) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

        nat_probs = F.softmax(logits, dim=1)

        true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()

        loss_robust = (1.0 / batch_size) * torch.sum(
            torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
        loss = loss_adv + float(beta) * loss_robust

        loss.backward()
        delta_grad = delta.grad.detach() 
        grad_mag += torch.sum(delta_grad.abs()*std)
        optimizer.step()
        train_loss += loss.item() * target.size(0)
        train_n += target.size(0)

    return train_acc, train_loss, train_n, grad_mag


def adjust_learning_rate(optimizer, epoch, total):
    """decrease the learning rate"""
    lr = args.lr_max
    if epoch >= (total/5*4):
        lr = args.lr_max * 0.01
    elif epoch >= (total/2):
        lr = args.lr_max * 0.1

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
    elif args.model == 'dense':
        model = DenseNet121().cuda()
    elif args.model == 'wide':
        model = WideResNet(34, 10, widen_factor=args.wide_factor, dropRate=0.0)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    train_time = 0
    highest_acc = 0
    highest_idx = 0
    num_steps = 0
    logger.info('Epoch \t Seconds \t LR \t BP\t Train Loss \t Train Acc \t Val Acc \t MG \t PGD Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        # adjust learning rate for SGD
        cur_lr = adjust_learning_rate(optimizer, epoch, args.epochs)

        # adversarial training



        train_acc, train_loss, train_n, grad_mag  = train(args, model, device, train_loader, num_steps, optimizer, epoch)
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
        # logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f\t %.4f',
        #     epoch, epoch_time, cur_lr, train_loss/train_n, val_acc,
        #     grad_mag, val_adv_acc)
        logger.info(f'{epoch}\t{epoch_time:.1f}\t{cur_lr:.4f}\t{num_steps+1:d}\t{train_loss/train_n:.4f}\t{train_acc/train_n:.4f}\t{val_acc:.4f}\t{grad_mag:.4f}\t{val_adv_acc:.4f}')

        if epoch == 0:
            num_steps += 1
            step_size = epsilon / num_steps
            if step_size < alpha:
                step_size = alpah
        elif epoch == 1:
            increase_threshold = args.gamma * grad_mag
        # elif ((grad_mag > increase_threshold and num_steps < args.num_steps)
        #         or (step_size * num_steps < args.epsilon)):
        elif (grad_mag > increase_threshold and num_steps < args.num_steps):
            increase_threshold = args.gamma * grad_mag
            num_steps += 1
            step_size = epsilon / num_steps
            if step_size < alpha:
                step_size = alpah

        if val_adv_acc > highest_acc and args.save_model:
            highest_acc = val_adv_acc
            highest_idx = epoch
            torch.save(model.state_dict(), os.path.join(args.out_dir, f'model_{args.model}.pth'))
    logger.info('Total train time: %.4f minutes', (train_time)/60)
    logger.info(f'Best checkpoint at {highest_idx}, {highest_acc}')
if __name__ == '__main__':
    main()