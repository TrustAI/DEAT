import os
import argparse
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import numpy as np
import logging
from preact_resnet import PreActResNet18
from wideresnet import WideResNet
from earlystop import earlystop
from utils import *


parser = argparse.ArgumentParser(description='PyTorch Friendly Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W')
parser.add_argument('--model', default='pre', type=str, choices=['pre', 'wide'])
parser.add_argument('--wide-factor', default=10, type=int, help='Widen factor')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=8, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=2, help='step size')
parser.add_argument('--normalization', default='std', type=str, choices=['std', '01','+-1'])
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')
parser.add_argument('--tau', type=int, default=3, help='step tau')
parser.add_argument('--beta',type=float,default=6.0,help='regularization parameter')
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")
parser.add_argument('--omega', type=float, default=0.0, help="random sample parameter for adv data generation")
parser.add_argument('--dynamictau', type=bool, default=False, help='whether to use dynamic tau')
parser.add_argument('--fname', default='output', type=str)
parser.add_argument('--data-dir', default='/mnt/storage0_8/torch_datasets/cifar-data', type=str)
parser.add_argument('--out-dir', default='fat_trade_out', type=str, help='Output directory')
parser.add_argument('--resume', type=str, default='', help='whether to resume training, default: None')
parser.add_argument('--save-model', action='store_true')
args = parser.parse_args()

# training settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
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
def TRADES_loss(adv_logits, natural_logits, target, beta):
    # Based on the repo TREADES: https://github.com/yaodongyu/TRADES
    batch_size = len(target)
    criterion_kl = nn.KLDivLoss(size_average=False).cuda()
    loss_natural = nn.CrossEntropyLoss(reduction='mean')(natural_logits, target)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                         F.softmax(natural_logits, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def train(model, train_loader, optimizer, tau):
    start_epoch_time = time.time()
    train_loss = 0
    train_n = 0
    bp_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        # Get friendly adversarial training data via early-stopped PGD
        output_adv, output_target, output_natural, count = earlystop(model, data, target, step_size=step_size,
                                                                     epsilon=epsilon, perturb_steps=args.num_steps,
                                                                     tau=tau, randominit_type="normal_distribution_randominit", loss_fn='kl', 
                                                                     mu=mu, std=std, rand_init=args.rand_init, omega=args.omega)
        bp_count += count
        model.train()
        optimizer.zero_grad()

        natural_logits = model(normalize(output_natural,mu,std))
        adv_logits = model(normalize(output_adv,mu,std))

        # calculate TRADES adversarial training loss
        loss = TRADES_loss(adv_logits,natural_logits,output_target,args.beta)

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * target.size(0)
        train_n += target.size(0)

    end_epoch_time = time.time()
    epoch_time = end_epoch_time - start_epoch_time

    return epoch_time, train_loss/train_n, bp_count/train_n

def adjust_tau(epoch, dynamictau):
    tau = args.tau
    if dynamictau:
        if epoch <= 50:
            tau = 0
        elif epoch <= 90:
            tau = 1
        else:
            tau = 2
    return tau


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 25:
        lr = args.lr * 0.1
    if epoch >= 40:
        lr = args.lr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
logfile = os.path.join(args.out_dir, args.fname+'.log')
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=logfile,
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO)
logger.info(args)
train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
if args.model == 'pre':
        model = PreActResNet18().cuda()
elif args.model == 'wide':
    model = WideResNet(34, 10, widen_factor=args.wide_factor, dropRate=0.0)
model = torch.nn.DataParallel(model).cuda()
model.train()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


train_time = 0
highest_acc = 0
highest_idx = 0
logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t #BP \t Val Acc \t PGD Acc')
for epoch in range(args.epochs):
    cur_lr = adjust_learning_rate(optimizer, epoch + 1)
    epoch_time, train_loss, nb_bp = train(model, train_loader, optimizer, adjust_tau(epoch + 1, args.dynamictau))
    train_time += epoch_time

    # Evaluation
    if args.model == 'pre':
        model_test = PreActResNet18().cuda()
    elif args.model == 'wide':
        model_test = WideResNet(34, 10, widen_factor=args.wide_factor, dropRate=0.0)
    model_test = torch.nn.DataParallel(model_test).cuda()
    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    val_adv_loss, val_adv_acc = evaluate_pgd(test_loader, model_test, mu, std, 10, 1, val=20, use_CWloss=True)
    val_loss, val_acc = evaluate_standard(test_loader, model_test, mu, std, val=20)
    logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %d \t %.4f \t %.4f',
        epoch, epoch_time, cur_lr, train_loss,nb_bp+1, val_acc, val_adv_acc)

    if val_adv_acc > highest_acc and args.save_model:
        highest_acc = val_adv_acc
        highest_idx = epoch
        torch.save(model.state_dict(), os.path.join(args.out_dir, f'model_{args.model}.pth'))
logger.info('Total train time: %.4f minutes', (train_time)/60)
logger.info(f'Best checkpoint at {highest_idx}, {highest_acc}')


