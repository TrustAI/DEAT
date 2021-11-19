import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),])

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),])
    
    num_workers = 2
    
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,)
    return train_loader, test_loader

def clean_evaluate(test_loader, model, random_init=False):
    total_loss = 0
    total_acc = 0
    n = 0
    epsilon = (8 / 255.) / std
    model.eval()
    with torch.no_grad():
        for _, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()

            delta = torch.zeros_like(X).cuda()
            if random_init:
                for i in range(len(epsilon)):
                    # delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
                    delta[:, i, :, :].normal_(0, epsilon[i][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            
            output = model(X+delta)
            loss = F.cross_entropy(output, y)
            total_loss += loss.item() * y.size(0)
            total_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return total_loss/n, total_acc/n