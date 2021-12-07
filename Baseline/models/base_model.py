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
import os, pdb
import argparse
import copy
import sys
 
sys.path.insert(0, '.')
from common import *
sys.path.insert(0, '../utils/')
from utils.helpers import *
# from common_SA import *
    
        
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 64
        self.res3 = resnet.layer2 # 1/8, 128
        self.res4 = resnet.layer3 # 1/16, 256
        self.res5 = resnet.layer4 # 1/32, 512

        ####################
        # freeze_BN(self)
        ####################

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, frames): # B,T,C,H,W
        B,C,H,W = frames.size()

        f = (frames - self.mean) / self.std

        x = self.conv1(f)
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 64
        r3 = self.res3(r2) # 1/8, 128
        r4 = self.res4(r3) # 1/16, 256
        r5 = self.res5(r4) # 1/32, 512
        
        return r5, r4, r3, r2, c1
    
    
class CrossAtt(nn.Module):
    def __init__(self, vis_dim=2048, lang_dim=1000, head_num=8, emb=512):
        super(CrossAtt, self).__init__()
    
        self.convSA = ConvSA(vis_dim, emb)
        self.linearSA = LinearSA(lang_dim, emb)
        C = emb+emb
        
        self.Query = nn.Conv2d(C, emb, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Key = nn.Conv2d(C, emb, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value = nn.Conv2d(C, emb, kernel_size=(3,3), padding=(1,1), stride=1)
        self.resValue = nn.Conv2d(C, emb, kernel_size=(3,3), padding=(1,1), stride=1)
        
        init_He(self)
        self.head_num = head_num
        
 
    def forward(self, vis, emb):  # vis: (B,C,H,W), emb: (B, L, C')
        
        B, C_v, H, W = vis.size()
        B, L, C_l = emb.size()
        
        vis_att = self.convSA(vis) # B,emb,H,W
        lang_att = self.linearSA(emb).transpose(1, 2)  # B,emb,L
        
        vis_ = vis_att.unsqueeze(2).repeat(1, 1, L, 1, 1)
        emb_ = lang_att.unsqueeze(3).unsqueeze(3).repeat(1, 1, 1, H, W)
        multi = torch.cat((vis_, emb_), 1) # B,C',L,H,W
        multi = multi.transpose(1, 2).reshape(B*L, -1, H, W)
        
        query = self.Query(multi).view(B, L, -1, H, W)
        key = self.Key(multi).view(B, L, -1, H, W)
        value = self.Value(multi).view(B, L, -1, H, W)
        res_value = self.resValue(multi).view(B, L, -1, H, W)
        
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        res_value = res_value.transpose(1, 2)
        
        query_ = query.reshape(B*self.head_num, -1, L*H*W) 
        key_ = key.reshape(B*self.head_num, -1, L*H*W)
        value_ = value.reshape(B*self.head_num, -1, L*H*W)
        
        att = torch.bmm(query_.transpose(1, 2), key_)
        att = F.softmax(att, dim=2)
        
        v_att = torch.bmm(value_, att.transpose(1, 2))
        v_att = v_att.reshape(B, -1, L, H, W)
        
        value_att = torch.mean(v_att + res_value, 2) 
        return value_att         

  
    
class Decoder(nn.Module):
    def __init__(self, mdim=256, multi_dim=512):
        super(Decoder, self).__init__()
        self.res_multi = nn.Sequential(
            nn.Conv2d(multi_dim, mdim, kernel_size=(3,3), padding=(1,1), stride=1),
            ResBlock(mdim, mdim)
        )
        
        self.RF4 = Refine(1024, mdim) 
        self.RF3 = Refine(512, mdim) 
        self.RF2 = Refine(256, mdim) 
        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, multi5, v5, v4, v3, v2, v1):
        m5 = self.res_multi(multi5)
        r4 = self.RF4(v4, m5) 
        r3 = self.RF3(v3, r4) 
        r2 = self.RF2(v2, r3) 
        
        pred = F.interpolate(r2, scale_factor=4, mode='bilinear', align_corners=False)
        return pred
        
    

class Mask(nn.Module):
    def __init__(self, dict_size=12099):
        super(Mask, self).__init__()
        self.name = "Baseline" 
        self.encoder = Encoder()
        
        self.embs = nn.ModuleList([nn.Embedding(dict_size, 1000)])
        self.cas = nn.ModuleList([CrossAtt(vis_dim=2048, lang_dim=1000)])
        
        self.decoder = Decoder()
 
    def forward(self, prev_frames, prev_masks, in_frames, words, eval=False):     
        
#         B,_,H,W = in_frames.size()

        embed = self.embs[0](words)
        vis_r5s, vis_r4s, vis_r3s, vis_r2s, vis_c1s = self.encoder(in_frames)
        multi_r5s = self.cas[0](vis_r5s, embed)
        
        logit = self.decoder(multi_r5s, vis_r5s, vis_r4s, vis_r3s, vis_r2s, vis_c1s)
        mask = F.softmax(logit, dim=1)
            
        return mask, logit

