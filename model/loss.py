import torch
eps = 1e-8


def criterion(output, target):
    g_loss = -torch.mean(torch.log(torch.abs(target - output) + eps))
    return g_loss


def D_loss(out_fake, out_real):
    d_loss = -torch.mean(torch.log(out_real + eps) + torch.log(1 - out_fake + eps))
    return d_loss

def G_loss(out_fake):
    g_loss = -torch.mean(torch.log(out_fake + eps))
    return g_loss
