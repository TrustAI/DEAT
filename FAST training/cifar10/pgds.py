import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

def f_6(outputs, y):

    label_mask = nn.functional.one_hot(y, 10).to(torch.bool)
    label_logit = outputs[label_mask]
    others = outputs[~label_mask].reshape(-1, 9)
    top_other_logit, _ = torch.max(others, dim=1)

    margin_w = torch.sum(top_other_logit - label_logit)

    return margin_w

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, random_init=True):
    
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if random_init:
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def attack_cw(model, X, y, epsilon, alpha, attack_iters, restarts, random_init=True):
    
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if random_init:
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = f_6(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y).detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def evaluate_pgd(test_loader, model, attack_iters, restarts,
                 eps=8, adv_radio=2, random_init=True, loss='ce', quick=None):

    epsilon = (eps / 255.) / std
    if attack_iters == 1:
        alpha = (eps / 255.) / std
    else:
        alpha = (adv_radio / 255.) / std

    if loss == 'ce':
        attack = attack_pgd
        adv_loss = nn.CrossEntropyLoss()
    elif loss == 'cw':
        attack = attack_cw
        adv_loss = f_6

    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack(model, X, y, epsilon, alpha, attack_iters, restarts, random_init)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = adv_loss(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
        if quick and i == quick - 1:
            break
    return pgd_loss/n, pgd_acc/n