import cv2, pdb
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from utils.helpers import *
import torch.nn.functional as F


def IoU(pred, gt, threshold=0.5, mean=True):
    #input: [b, h, w]
    #target: [b, h, w] long

    x = (pred > threshold).float()
    y = gt.float()

    I = x * y
    U = (x + y) - I
    n_I = I.sum(-1).sum(-1) # sum over spatial dims
    n_U = U.sum(-1).sum(-1)

    iou = n_I / (n_U + 1e-5)
    # mean over mini-batch
    if mean:
        return torch.mean(iou).item()
    else:
        return 


def mIoU(input, target, return_batch=False):
    # now only support binary
    # _, input = torch.max(input, dim=1) # for mo
    x = (input > 0.5).float()
    y = (target > 0.5).float()

    i = torch.sum(torch.sum(x*y, dim=-1), dim=-1) # sum over spatial dims
    u = torch.sum(torch.sum((x+y)-(x*y), dim=-1), dim=-1) 

    iou = i / (u + 1e-4) # b
    if return_batch:
        return iou
    else:
        # mean over mini-batch
        return torch.mean(iou).item()    

def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array
    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
