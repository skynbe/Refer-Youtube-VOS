from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo

from torchvision import models

# general libs
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import copy
# from itertools impor
import cv2
import random, pdb

import logging
import logging.handlers
from datetime import datetime

##########################################
############   Generic   #################
##########################################



def time_weight_same(N):
    loss_weight = np.ones((N,))
    loss_weight[0] = 0.
    loss_weight /= np.sum(np.array(loss_weight))
    return loss_weight.tolist()



##########################################
############   pytorch   #################
##########################################
def my_load_state_dict(model, state_dict):
    # version 2: support tensor same name but size is different
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            print('[Warning] Found key "{}" in file, but not in current model'.format(name))
    
    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        print('[Warning] Cant find keys "{}" in file'.format(missing))

def noDP(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def load_NoPrefix(path, length):
    # load dataparallel wrapped model properly
    state_dict = torch.load(path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[length:] # remove `Scale.`
        new_state_dict[name] = v
    return new_state_dict

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

def Dilate(module, rate):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (3,3) and m.stride == (2,2):  # first conv with striding
                m.stride = (1,1)
            elif m.kernel_size == (1,1) and m.stride == (2,2):  # non-identity skip for first block
                m.stride = (1,1)
            elif m.kernel_size == (3,3) and m.stride == (1,1):  # normal 3x3 conv
                m.dilation = (rate,rate)
                m.padding = (rate,rate)
            elif m.kernel_size == (1,1) and m.stride == (1,1):  # 1x1 compressor
                pass
            else:
                print('wtf:', m)


def ToCudaVariable(xs, volatile=False, requires_grad=False):
    if torch.cuda.is_available():
        return [Variable(x.cuda(), volatile=volatile, requires_grad=requires_grad) for x in xs]
    else:
        return [Variable(x, volatile=volatile, requires_grad=requires_grad) for x in xs]

def ToCuda(xs):
    if torch.cuda.is_available():
        if isinstance(xs, list) or isinstance(xs, tuple):
            return [x.cuda() for x in xs]
        else:
            return xs.cuda()
    else:
        return xs
    
    
def ExpandTime_(tensor, T):
    if tensor is None:
        return None
    BT,C,H,W = tensor.size()
    assert BT % T == 0
    B = int(BT/T)
    # Tensor : B*T,C,H,W
    return tensor.reshape(B, T, *tensor.size()[1:])

def FoldTime_(tensor):
    if tensor is None:
        return None
    B,T,C,H,W = tensor.size()
    return tensor.reshape(-1, *tensor.size()[2:])


def FoldTime(tensor):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        return [FoldTime_(t) for t in tensor]
    else:
        return FoldTime_(tensor)

def ExpandTime(tensor, T):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        return [ExpandTime_(t, T) for t in tensor]
    else:
        return ExpandTime_(tensor, T)

    
    

##########################################
###########   Interaction   ##############
##########################################

def Click(E, M, r, th_iou, rnd):
    def _sample(mask, rnd):
        idxs = np.where(mask>0.5)
        num_pixel = len(idxs[0])
        if num_pixel == 0:
            return -1, -1
        t = rnd.randint(0, num_pixel-1)
        return idxs[0][t], idxs[1][t]

    biou = mIoU(E, M, return_batch=True).cpu().numpy()
    npE = torch.round(E.detach()).cpu().numpy()
    npM = M.detach().data.float().cpu().numpy()
    size = npE.shape[1:]
    P = np.zeros_like(npE)
    N = np.zeros_like(npE)
    
    py, px, nx, ny = -1, -1, -1, -1
    if r == 1:
        for b in range(npE.shape[0]):
            py, px = _sample(npM[b], rnd)
            if py > 0 and px > 0:  # positive
                P[b] = cstm_normalize(make_gaussian(size, sigma=10, center=(px, py)), 1)
    else:
        fp = npE * (1-npM) # false positive: E = 1 / M = 0
        fn =  (1-npE) * npM # false negative: E = 0 / M = 1
        for b in range(npE.shape[0]):
            iou = db_eval_iou(npM[b],npE[b])
            if iou < th_iou: 
                ratio = np.sum(fn[b]) / (np.sum(fn[b]) + np.sum(fp[b]))
                if rnd.uniform(0,1) < ratio: # if FN is large
                    py, px = _sample(fn[b], rnd)
                    if py > 0 and px > 0:  # positive
                        P[b] = cstm_normalize(make_gaussian(size, sigma=10, center=(px, py)), 1)
                else:
                    ny, nx = _sample(fp[b], rnd)
                    if ny > 0 and nx > 0:  # negative
                        N[b] = cstm_normalize(make_gaussian(size, sigma=10, center=(nx, ny)), 1)

    return torch.from_numpy(P).cuda(), torch.from_numpy(N).cuda()


'''
Code from DEXTR:
https://github.com/scaelles/DEXTR-PyTorch/blob/master/dataloaders/helpers.py
'''
def make_gaussian(size, sigma=10, center=None, d_type=np.float32):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)


def cstm_normalize(im, max_value):
    """
    Normalize image to range 0 - max_value
    """
    imn = max_value*(im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn



##########################################
###########   DAVIS measures   ###########
##########################################
# https://github.com/fperazzi/davis-2017/tree/master/python/lib/davis/measures

def db_eval_iou(annotation,segmentation):
    """ Compute region similarity as the Jaccard Index.

    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.

    Return:
        jaccard (float): region similarity

    """

    annotation   = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)

def db_eval_boundary(foreground_mask,gt_mask,bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.

    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    foreground_mask = (foreground_mask>0.5).cpu().numpy().transpose(1, 2, 0)
    gt_mask = gt_mask.cpu().numpy().transpose(1, 2, 0)
    
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask)[:, :, 0]
    gt_boundary = seg2bmap(gt_mask)[:, :, 0]
    
    from skimage.morphology import binary_dilation,disk

    fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg     = np.sum(fg_boundary)
    n_gt     = np.sum(gt_boundary)
    

    #% Compute precision and recall
    if n_fg == 0 and  n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0  and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match)/float(n_fg)
        recall    = np.sum(gt_match)/float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2*precision*recall/(precision+recall)

    return F

def seg2bmap(seg,width=None,height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
        seg     : Segments labeled from 1..k.
        width      :    Width of desired bmap  <= seg.shape[1]
        height  :    Height of desired bmap <= seg.shape[0]

    Returns:
        bmap (ndarray):    Binary boundary map.

     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(np.bool)
    seg[seg>0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width  = seg.shape[1] if width  is None else width
    height = seg.shape[0] if height is None else height

    h,w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
            'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

    e  = np.zeros_like(seg)
    s  = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:,:-1]    = seg[:,1:]
    s[:-1,:]    = seg[1:,:]
    se[:-1,:-1] = seg[1:,1:]

    b        = seg^e | seg^s | seg^se
    b[-1,:]  = seg[-1,:]^e[-1,:]
    b[:,-1]  = seg[:,-1]^s[:,-1]
    b[-1,-1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height,width))
        for x in range(w):
            for y in range(h):
                if b[y,x]:
                    j = 1+floor((y-1)+height / h)
                    i = 1+floor((x-1)+width  / h)
                    bmap[j,i] = 1

    return bmap




##########################################
###########   visualization   ############
##########################################
### https://github.com/albertomontesg/davis-interactive/blob/71b0c12bbb8a0c765a888fbd325b226f2c51daf7/davisinteractive/utils/visualization.py

def _pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def overlay_mask(im, ann, alpha=0.5, colors=None, contour_thickness=None):
    """ Overlay mask over image.
    This function allows you to overlay a mask over an image with some
    transparency.
    # Arguments
        im: Numpy Array. Array with the image. The shape must be (H, W, 3) and
            the pixels must be represented as `np.uint8` data type.
        ann: Numpy Array. Array with the mask. The shape must be (H, W) and the
            values must be intergers
        alpha: Float. Proportion of alpha to apply at the overlaid mask.
        colors: Numpy Array. Optional custom colormap. It must have shape (N, 3)
            being N the maximum number of colors to represent.
        contour_thickness: Integer. Thickness of each object index contour draw
            over the overlay. This function requires to have installed the
            package `opencv-python`.
    # Returns
        Numpy Array: Image of the overlay with shape (H, W, 3) and data type
            `np.uint8`.
    """
    im, ann = np.asarray(im, dtype=np.uint8), np.asarray(ann, dtype=np.int)
    if im.shape[:-1] != ann.shape:
        raise ValueError('First two dimensions of `im` and `ann` must match')
    if im.shape[-1] != 3:
        raise ValueError('im must have three channels at the 3 dimension')

    colors = colors or _pascal_color_map()
    colors = np.asarray(colors, dtype=np.uint8)

    mask = colors[ann]
    fg = im * alpha + (1 - alpha) * mask

    img = im.copy()
    img[ann > 0] = fg[ann > 0]

    if contour_thickness:  # pragma: no cover
        import cv2
        for obj_id in np.unique(ann[ann > 0]):
            contours = cv2.findContours((ann == obj_id).astype(
                np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            cv2.drawContours(img, contours[0], -1, colors[obj_id].tolist(),
                             contour_thickness)
    return img
