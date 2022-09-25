import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *

def mart_loss(model,
              x_natural,
              y,
              optimizer,
              step_size,
              epsilon,
              mu,std,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.train()
    batch_size = len(x_natural)
    # generate adversarial example
    # x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    # iter_delta = 0.001*torch.randn_like(x_natural).cuda()
    # iter_delta = clamp(iter_delta, lower_limit - x_natural, upper_limit - x_natural)
    # x_adv = x_natural.detach() + iter_delta
    if distance == 'l_inf':
        delta = 0.001*torch.randn_like(x_natural).cuda()
        delta = clamp(delta, lower_limit - x_natural, upper_limit - x_natural)
        for _ in range(perturb_steps):
            delta.requires_grad_()

            loss_ce = F.cross_entropy(model(normalize(x_natural+delta,mu,std)), y)
            loss_ce.backward()
            grad = delta.grad.detach()
            # x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            # x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            # x_adv = torch.clamp(x_adv, 0.0, 1.0)
            delta.data =  torch.clamp(delta + step_size * torch.sign(grad), min=-epsilon, max=epsilon)
            # delta.data = clamp(delta + step_size * torch.sign(grad), -epsilon, epsilon)
            delta.data[:x_natural.size(0)] = clamp(delta[:x_natural.size(0)], lower_limit - x_natural, upper_limit - x_natural)
            delta.grad.zero_()
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = x_natural.detach() + delta.detach()
    # zero gradient
    optimizer.zero_grad()

    logits = model(normalize(x_natural,mu,std))

    logits_adv = model(normalize(x_adv,mu,std))

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust

    return loss

def mmart_loss(model,
              x_natural,
              y,
              optimizer,
              step_size,
              epsilon,
              mu,std,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.train()
    batch_size = len(x_natural)
    grad_mag = 0
    # generate adversarial example
    # x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    # iter_delta = 0.001*torch.randn_like(x_natural).cuda()
    # iter_delta = clamp(iter_delta, lower_limit - x_natural, upper_limit - x_natural)
    # x_adv = x_natural.detach() + iter_delta
    if distance == 'l_inf':
        delta = 0.001*torch.randn_like(x_natural).cuda()
        delta = clamp(delta, lower_limit - x_natural, upper_limit - x_natural)
        for rr in range(perturb_steps):
            delta.requires_grad_()

            loss_ce = F.cross_entropy(model(normalize(x_natural+delta,mu,std)), y)
            loss_ce.backward()
            grad = delta.grad.detach()
            # x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            # x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            # x_adv = torch.clamp(x_adv, 0.0, 1.0)
            delta.data =  torch.clamp(delta + step_size * torch.sign(grad), min=-epsilon, max=epsilon)
            # delta.data = clamp(delta + step_size * torch.sign(grad), -epsilon, epsilon)
            delta.data[:x_natural.size(0)] = clamp(delta[:x_natural.size(0)], lower_limit - x_natural, upper_limit - x_natural)
            delta.grad.zero_()
            # if rr + 1 < perturb_steps:
            #     continue
            # grad_mag += torch.sum(grad.abs()*std)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = x_natural.detach() + delta.detach()
    # zero gradient
    optimizer.zero_grad()

    logits = model(normalize(x_natural,mu,std))

    logits_adv = model(normalize(x_adv,mu,std))

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust

    return loss