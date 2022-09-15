import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from wideresnet import WideResNet
from preact_resnet import PreActResNet18
from vgg import VGG16,VGG19
from densenet import DenseNet121
from autoattack import AutoAttack


def get_test_loader(dir_, batch_size):
    
    num_workers = 2
    
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,)
    return test_loader

def evaluate_autoattack(test_loader, model, batch_size, eps=8, log=None):
    epsilon = (eps / 255.)
    adversary = AutoAttack(model, norm='Linf', eps=epsilon, log_path=log, version='standard')
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        x_adv = adversary.run_standard_evaluation(X, y, bs=batch_size)


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=200, type=int)
parser.add_argument('--normalization', default='std', type=str, choices=['std', '01','+-1'])
parser.add_argument('--data-dir', default='/mnt/storage0_8/torch_datasets/cifar-data', type=str)
parser.add_argument('--model', default='pre', type=str, choices=['pre', 'wide', 'vgg16', 'vgg19', 'dense'])
parser.add_argument('--model-dir', default='mdeat_out', type=str)
parser.add_argument('--model-name', default='model_pre', type=str)
parser.add_argument('--log-name', default='aa_score', type=str)
args = parser.parse_args()

if args.normalization == 'std':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
elif args.normalization == '01':
    mean = (0, 0, 0)
    std = (1, 1, 1)
elif args.normalization == '+-1':
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

log_path = os.path.join(args.model_dir,args.log_name+'.log')
test_loader = get_test_loader(args.data_dir, args.batch_size)
model_path = os.path.join(args.model_dir,args.model_name+'.pth')
checkpoint = torch.load(model_path)
if args.model == 'pre':
    net = PreActResNet18().cuda()
elif args.model == 'vgg19':
    net = VGG19().cuda()
elif args.model == 'vgg16':
    net = VGG16().cuda()
elif args.model == 'dense':
    net = DenseNet121().cuda()
net = torch.nn.DataParallel(net).cuda()
net.load_state_dict(checkpoint)
model_test = nn.Sequential(Normalize(mean=mean, std=std), net)
model_test.float()
model_test.eval()
print(f'Evaluating {model_path}')
evaluate_autoattack(test_loader,model_test,args.batch_size,8,log_path)