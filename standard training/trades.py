import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from utils import *

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size,
                epsilon,
                mu,std,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.train()
    batch_size = len(x_natural)
    # generate adversarial example
    if distance == 'l_inf':
        delta = 0.001 * torch.randn(x_natural.shape).cuda()
        delta.requires_grad = True
        for _ in range(perturb_steps):
            # with torch.enable_grad():
            loss_kl = criterion_kl(
                        F.log_softmax(model(normalize(x_natural+delta,mu,std)), dim=1),
                        F.softmax(model(normalize(x_natural,mu,std)), dim=1))
            loss_kl.backward()
            grad = delta.grad.detach()
            delta.data =  torch.clamp(delta + step_size * torch.sign(grad), min=-epsilon, max=epsilon)
            delta.data[:x_natural.size(0)] = clamp(delta[:x_natural.size(0)], lower_limit - x_natural, upper_limit - x_natural)
            delta.grad.zero_()
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(
                                F.log_softmax(model(normalize(adv,mu,std)), dim=1),
                                F.softmax(model(normalize(x_natural,mu,std)), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    # x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(normalize(x_natural,mu,std))
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(
                                        F.log_softmax(model(normalize(x_natural+delta,mu,std)), dim=1),
                                        F.softmax(model(normalize(x_natural,mu,std)), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def mtrades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size,
                epsilon,
                mu,std,
                perturb_steps=10,
                beta=1.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.train()
    batch_size = len(x_natural)

    grad_mag = 0
    delta = 0.001 * torch.randn(x_natural.shape).cuda()
    delta.requires_grad = True
    for rr in range(perturb_steps):
        # with torch.enable_grad():
        loss_kl = criterion_kl(
                    F.log_softmax(model(normalize(x_natural+delta,mu,std)), dim=1),
                    F.softmax(model(normalize(x_natural,mu,std)), dim=1))
        loss_kl.backward()
        grad = delta.grad.detach()
        delta.data =  torch.clamp(delta + step_size * torch.sign(grad), min=-epsilon, max=epsilon)
        delta.data[:x_natural.size(0)] = clamp(delta[:x_natural.size(0)], lower_limit - x_natural, upper_limit - x_natural)
        if rr + 1 < perturb_steps:
            delta.grad.zero_()
            continue
        grad_mag += torch.sum(grad.abs()*std)
    
    model.train()

    # x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(normalize(x_natural,mu,std))
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(
                                        F.log_softmax(model(normalize(x_natural+delta,mu,std)), dim=1),
                                        F.softmax(model(normalize(x_natural,mu,std)), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss, grad_mag