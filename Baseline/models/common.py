from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
 
# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import sys

sys.path.insert(0, '../utils/')
from utils.helpers import *

##########################################
############   Generic   #################
##########################################

def overlay_box(sem, yxhw, color=[1,0,0]):
    # box
    sem = sem.copy()
    size = (256,256)
    ymin = int(np.clip(round(yxhw[0] - (yxhw[2]-1)/2.), 0, size[0]-1))
    ymax = int(np.clip(round(yxhw[0] + (yxhw[2]-1)/2.), 0, size[0]-1))
    xmin = int(np.clip(round(yxhw[1] - (yxhw[3]-1)/2.), 0, size[1]-1))
    xmax = int(np.clip(round(yxhw[1] + (yxhw[3]-1)/2.), 0, size[1]-1))
 
    sem[:, ymin:ymin+1, xmin:xmax+1] = np.reshape(np.array(color), [3, 1, 1])
    sem[:, ymax:ymax+1, xmin:xmax+1] = np.reshape(np.array(color), [3, 1, 1])
    sem[:, ymin:ymax+1, xmin:xmin+1] = np.reshape(np.array(color), [3, 1, 1])
    sem[:, ymin:ymax+1, xmax:xmax+1] = np.reshape(np.array(color), [3, 1, 1])
 
    return sem
 
def save_result(path, f1, p1, e1, m1, eb, gb, size=[256,256], n=0):
    f1 = (f1.data[n]).cpu().numpy()
    p1 = (p1.data[n]).cpu().numpy()
    e1 = (e1.data[n]).cpu().numpy()
    m1 = (m1.data[n]).cpu().numpy()
    eb = (eb.data[n]).cpu().numpy()
    gb = (gb.data[n]).cpu().numpy()
 
 
    canvas = np.zeros((3, 2*size[0], 2*size[1]))
 
    canvas[:,0*size[0]:1*size[0],0*size[1]:1*size[1]] = f1
    canvas[:,0*size[0]:1*size[0],1*size[1]:2*size[1]] = np.stack([p1, p1, p1])
    sm = np.stack([m1, m1, m1])
    sm = overlay_box(sm, gb, color=[0,1,0])
    sm = overlay_box(sm, eb, color=[1,0,0])
    sem = np.stack([e1, e1, e1])
    sem = overlay_box(sem, gb, color=[0,1,0])
    sem = overlay_box(sem, eb, color=[1,0,0])
    canvas[:,1*size[0]:2*size[0],0*size[1]:1*size[1]] = sm
    canvas[:,1*size[0]:2*size[0],1*size[1]:2*size[1]] = sem
 
    canvas = np.transpose(canvas, [1,2,0])
 
    im = Image.fromarray((canvas * 255.).astype(np.uint8))
    im.save(path)



##########################################
############   LOSS      #################
##########################################

def BoxIOULoss(pred, gt):
    pr_ymin, pr_ymax, pr_xmin, pr_xmax = yxhw2minmax(pred)
    gt_ymin, gt_ymax, gt_xmin, gt_xmax  = yxhw2minmax(gt)
 
    I_w = torch.clamp(torch.min(pr_xmax, gt_xmax) - torch.max(pr_xmin, gt_xmin), min=0, max=99999)
    I_h = torch.clamp(torch.min(pr_ymax, gt_ymax) - torch.max(pr_ymin, gt_ymin), min=0, max=99999)
    I_area = I_w * I_h
    gt_area = (gt_ymax - gt_ymin) * (gt_xmax - gt_xmin)
    pr_area = (pr_ymax - pr_ymin) * (pr_xmax - pr_xmin)
 
    iou = I_area / (gt_area + pr_area - I_area + 1e-4)
    return 1 - iou




##########################################
############   Ops       #################
##########################################

def get_ROI_grid(roi, src_size, dst_size, scale=1.):
    # scale height and width
    ry, rx, rh, rw = roi[:,0], roi[:,1], scale * roi[:,2], scale * roi[:,3]
        
    # convert ti minmax  
    ymin = ry - rh/2.
    ymax = ry + rh/2.
    xmin = rx - rw/2.
    xmax = rx + rw/2.
        
    h, w = src_size[0], src_size[1] 
    # theta
    theta = ToCudaVariable([torch.zeros(roi.size()[0],2,3)])[0]
    theta[:,0,0] = (xmax - xmin) / (w - 1)
    theta[:,0,2] = (xmin + xmax - (w - 1)) / (w - 1)
    theta[:,1,1] = (ymax - ymin) / (h - 1)
    theta[:,1,2] = (ymin + ymax - (h - 1)) / (h - 1)

    #inverse of theta
    inv_theta = ToCudaVariable([torch.zeros(roi.size()[0],2,3)])[0]
    det = theta[:,0,0]*theta[:,1,1]
    adj_x = -theta[:,0,2]*theta[:,1,1]
    adj_y = -theta[:,0,0]*theta[:,1,2]
    inv_theta[:,0,0] = w / (xmax - xmin) 
    inv_theta[:,1,1] = h / (ymax - ymin) 
    inv_theta[:,0,2] = adj_x / det
    inv_theta[:,1,2] = adj_y / det
    # make affine grid
    fw_grid = F.affine_grid(theta, torch.Size((roi.size()[0], 1, dst_size[0], dst_size[1])))
    bw_grid = F.affine_grid(inv_theta, torch.Size((roi.size()[0], 1, src_size[0], src_size[1])))
    return fw_grid, bw_grid, theta, inv_theta
 
def box_affine_transform(b, inv_theta, src_size, dst_size):
    # box affine need inverse theta.
    ymin = (b[:,0] - b[:,2]/2. - (src_size[0]-1) /2.) / ((src_size[0]-1) /2.)
    ymax = (b[:,0] + b[:,2]/2. - (src_size[0]-1) /2.) / ((src_size[0]-1) /2.)
    xmin = (b[:,1] - b[:,3]/2. - (src_size[1]-1) /2.) / ((src_size[1]-1) /2.)
    xmax = (b[:,1] + b[:,3]/2. - (src_size[1]-1) /2.) / ((src_size[1]-1) /2.)

    n_ymin = (ymin * inv_theta[:,1,1] + inv_theta[:,1,2]) * (dst_size[0]-1) /2. + (dst_size[0]-1) /2.
    n_ymax = (ymax * inv_theta[:,1,1] + inv_theta[:,1,2]) * (dst_size[0]-1) /2. + (dst_size[0]-1) /2.
    n_xmin = (xmin * inv_theta[:,0,0] + inv_theta[:,0,2]) * (dst_size[1]-1) /2. + (dst_size[1]-1) /2.
    n_xmax = (xmax * inv_theta[:,0,0] + inv_theta[:,0,2]) * (dst_size[1]-1) /2. + (dst_size[1]-1) /2.

    new_y = (n_ymax + n_ymin)/2.
    new_x = (n_xmax + n_xmin)/2.
    new_h = n_ymax - n_ymin
    new_w = n_xmax - n_xmin  

    new_b = torch.stack([new_y, new_x, new_h, new_w], dim=1)

    return new_b

def mask2yxhw(mask, scale, min_size, clip):
    np_mask = mask.data.cpu().numpy()
    np_yxhw = np.zeros((np_mask.shape[0], 4), dtype=np.float32)
    for b in range(np_mask.shape[0]):
        mys, mxs = np.where(np_mask[b] >= 0.49)
        all_ys = np.concatenate([mys])
        all_xs = np.concatenate([mxs])

        if all_ys.size == 0 or all_xs.size == 0:
            ymin, ymax = 0, np_mask.shape[1]-1
            xmin, xmax = 0, np_mask.shape[2]-1
        else:
            ymin, ymax = np.min(all_ys), np.max(all_ys)
            xmin, xmax = np.min(all_xs), np.max(all_xs)

        # apply scale
        orig_h = ymax - ymin + 1
        orig_w = xmax - xmin + 1
        ymin = ymin - (scale - 1) / 2. * orig_h
        ymax = ymax + (scale - 1) / 2. * orig_h
        xmin = xmin - (scale - 1) / 2. * orig_w
        xmax = xmax + (scale - 1) / 2. * orig_w

        # make sure minimum 32 pixel each side
        if (ymax-ymin) < min_size:
            res = min_size - (ymax-ymin)
            ymin -= int(res/2)
            ymax += int(res/2)
        if (xmax-xmin) < min_size:
            res = min_size - (xmax-xmin)
            xmin -= int(res/2)
            xmax += int(res/2)

        if clip:
            ymin = np.maximum(0, ymin)  
            ymax = np.minimum(np_mask.shape[1], ymax)    
            xmin = np.maximum(0, xmin)  
            xmax = np.minimum(np_mask.shape[2], xmax) 

        # final ywhw
        y = (ymax + ymin) / 2.
        x = (xmax + xmin) / 2.
        h = ymax - ymin + 1
        w = xmax - xmin + 1

        yxhw = np.array([y,x,h,w], dtype=np.float32)
        
        np_yxhw[b] = yxhw
        
    return ToCuda(torch.from_numpy(np_yxhw))



def is_there_box(mask):
    num_pixel = np.sum(mask.data.cpu().numpy(), axis=(1,2))
    yes = (num_pixel > 0).astype(np.float32)
    return ToCuda(torch.from_numpy(yes))

def yxhw2minmax(yxhw):
    y,x,h,w = yxhw[:,0], yxhw[:,1], yxhw[:,2], yxhw[:,3]
    # mm = torch.stack([y - h/2., y + h/2., x - w/2., x + w/2.], dim=1)
    return [y - h/2., y + h/2., x - w/2., x + w/2.]

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




##########################################
############   Module    #################
##########################################

def init_He(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def freeze_BN(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters():
                p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
        init_He(self)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, expansion):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes//expansion, kernel_size=1)
        self.conv2 = nn.Conv2d(inplanes//expansion, inplanes//expansion, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(inplanes//expansion, planes , kernel_size=1)
        if inplanes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(inplanes, planes, kernel_size=1, padding=0)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        r = self.conv3(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r 

class GlobalConv(nn.Module):
    def __init__(self, inplanes, planes, kh, kw):
        super(GlobalConv, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, planes, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))
        self.conv_l2 = nn.Conv2d(planes, planes, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r1 = nn.Conv2d(inplanes, planes, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r2 = nn.Conv2d(planes, planes, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))

    def forward(self, x):
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        return x

class GCBlock(nn.Module):
    def __init__(self, inplanes, planes, kh, kw, expansion):
        super(GCBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes//expansion, kernel_size=1, bias=False)
        self.conv2 = GlobalConv(inplanes//expansion, inplanes//expansion, kh, kw)
        self.conv3 = nn.Conv2d(inplanes//expansion, planes , kernel_size=1, bias=False)
        if inplanes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(inplanes, planes, kernel_size=1, padding=0)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        r = self.conv3(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r 


class NonLocal(nn.Module):
    def __init__(self, mdim):
        super(NonLocal, self).__init__()
        self.conv_th = nn.Conv2d(mdim, int(mdim/2), kernel_size=1, padding=0)
        self.conv_pi = nn.Conv2d(mdim, int(mdim/2), kernel_size=1, padding=0)
        self.conv_g = nn.Conv2d(mdim, int(mdim/2), kernel_size=1, padding=0)
        self.conv_out = nn.Conv2d(int(mdim/2), mdim, kernel_size=1, padding=0)
 
    def forward(self, x1, x2):
        res = x1
        e1 = self.conv_th(x1) 
        e1 = e1.view(-1, e1.size()[1], e1.size()[2]*e1.size()[3]) 
        e1 = torch.transpose(e1, 1, 2)  # b, hw1, c/2
 
        e2 = self.conv_pi(x2) 
        e2 = e2.view(-1, e2.size()[1], e2.size()[2]*e2.size()[3])  # b, c/2, hw2
 
        f = torch.bmm(e1, e2) # b, hw1, hw2
        f = F.softmax(f.view(-1, f.size()[1]*f.size()[2]), dim=1).view(-1, f.size()[1], f.size()[2]) # b, hw1, hw2
 
        g2 = self.conv_g(x2)  
        g2 = g2.view(-1, g2.size()[1], g2.size()[2]*g2.size()[3])
        g2 = torch.transpose(g2, 1, 2) # b, hw2, c/2
 
        out = torch.bmm(f, g2)  # b, hw1, c/2
        out = torch.transpose(out, 1, 2).contiguous() # b, c/2, hw1
        out = out.view(-1, out.size()[1], x1.size()[2], x1.size()[3]) # b, c/2, h1, w1
        out = self.conv_out(out)  # b, c, h1, w1
 
        out = out + res
        return out


##############################


class ConvSA(nn.Module):
    def __init__(self, C, emb=512):
        super(ConvSA, self).__init__()
        self.Query = nn.Conv2d(C, emb, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Key = nn.Conv2d(C, emb, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value = nn.Conv2d(C, emb, kernel_size=(3,3), padding=(1,1), stride=1)
        init_He(self)
 
    def forward(self, feat): # (B, C, H, W)
        B, C, H, W = feat.size()
        query = self.Query(feat)
        key = self.Key(feat)
        value = self.Value(feat)
        
        query_ = query.reshape(B, -1, H*W)
        key_ = key.reshape(B, -1, H*W)
        value_ = value.reshape(B, -1, H*W)
        
        att = torch.bmm(query_.transpose(1, 2), key_) # B, HW, HW
        att = F.softmax(att, dim=2) # normalize by key_HW
        
        v_att = torch.bmm(value_, att.transpose(1, 2)) # B, emb, HW
        v_att = v_att.view(B, -1, H, W)
        
        value_att = v_att + value # B, emb, H, W
        return value_att  
    
    
    
class LinearSA(nn.Module):
    
    def __init__(self, C, emb=512):
        super(LinearSA, self).__init__()
        self.Query = nn.Linear(C, emb)
        self.Key = nn.Linear(C, emb)
        self.Value = nn.Linear(C, emb)
        init_He(self)
 
    def forward(self, feat): # (B, L, C)
        B, L, C = feat.size()
        
        query = self.Query(feat)
        key = self.Key(feat)
        value = self.Value(feat)
        
        att = torch.bmm(query, key.transpose(1, 2)) # B, L, L
        att = F.softmax(att, dim=2) # normalize by key_L
        
        v_att = torch.bmm(att, value) # B, L, emb
        
        value_att = v_att + value # B, L, emb
        return value_att 
    
        
    
class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='nearest')
        m = self.ResMM(m)
        return m



class UpsampleBlock(nn.Module):
    def __init__(self, num_upsample):
        super(UpsampleBlock, self).__init__()

        if num_upsample > 0:
            blocks = []
            for _ in range(num_upsample):
                upblock = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='nearest')
                )
                blocks.append(upblock)
            self.blocks = nn.Sequential(*blocks)
        else:
            self.blocks = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        init_He(self)
 
    def forward(self, x):
        out = self.blocks(x)
        return out
    