import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from preact_resnet import PreActResNet18
from wideresnet import WideResNet
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=200, type=int)
parser.add_argument('--normalization', default='std', type=str, choices=['std', '01','+-1'])
parser.add_argument('--data-dir', default='/mnt/storage0_8/torch_datasets/cifar-data', type=str)
parser.add_argument('--model-dir', default='mdeat_out', type=str)
parser.add_argument('--model-name', default='model_pre', type=str)
parser.add_argument('--model', default='pre', type=str, choices=['pre', 'wide'])
parser.add_argument('--wide-factor', default=10, type=int, help='Widen factor')
args = parser.parse_args()

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
model_path = os.path.join(args.model_dir,args.model_name+'.pth')
checkpoint = torch.load(model_path)
if args.model == 'pre':
    model_test = PreActResNet18().cuda()
elif args.model == 'wide':
    model_test = WideResNet(34, 10, widen_factor=args.wide_factor, dropRate=0.0)
model_test = torch.nn.DataParallel(model_test).cuda()
model_test.load_state_dict(checkpoint)
model_test.float()
model_test.eval()
print(f'Evaluating {model_path}')
cw_loss, cw_acc = evaluate_pgd(test_loader, model_test, mu, std, 20, 1, use_CWloss=True)
pgd100_loss, pgd100_acc = evaluate_pgd(test_loader, model_test, mu, std, 100, 1, use_CWloss=False)
print('PGD100 Loss \t PGD100 Acc \t CW20 Loss \t CW20 Acc')
print('{:.4f} \t {:.4f} \t  {:.4f} \t  {:.4f}'.format(pgd20_loss, pgd20_acc, cw_loss, cw_acc))
