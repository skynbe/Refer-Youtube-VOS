# -*- coding: utf-8 -*-
from __future__ import division
import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import torch.nn as nn
from torch.utils import data
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2, pdb
from PIL import Image
import numpy as np
from tqdm import tqdm
from addict import Dict

import os, sys, logging, time, random, json
from pathlib import Path
from shutil import copyfile

import math

### My libs
sys.path.append('utils/')
sys.path.append('models/')
sys.path.append('dataset/')

from utils.helpers import *

from io_utils import *
from eval_utils import *

import dataset.factory as factory

DATA_ROOT = Path('./data')
CHECKPOINT_ROOT = Path('./checkpoint')
OUTPUT_IMG_ROOT = Path('./validation')
EVALUTAION_ROOT = Path('./evaluation')


class Trainer():

    def __init__(self, args):
        import importlib
        
        self.args = args
        
        self.init_lr = args.init_lr if args.init_lr else 1e-4
        self.lr = 0
        
        self.epoch = -1
        self.max_epoch = args.max_epoch
        self.decay_epochs = args.decay_epochs
        self.lr_decay = args.lr_decay
        self.save_every = 1 if not args.save_every else args.save_every
        
        self.img_size = (args.img_size, args.img_size)
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size
        self.max_N = 2 if not args.max_N else args.max_N
        self.max_skip = args.max_skip

        self.desc = args.desc
        self.arch = args.arch
        self.splits = args.splits
            
        self.model = importlib.import_module('models.{}'.format(self.arch)).Mask()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.scheme = self.base_scheme
        
        self.logger = get_logger(self.arch)
        
        
    def get_file_name(self):
        file_name = self.arch
        if self.desc:
            file_name += '_' + self.desc
        return file_name
    
    
    def get_save_path(self):
        file_name = self.get_file_name()
        save_path = CHECKPOINT_ROOT / self.dataset / file_name
        return save_path
            

    def cuda(self):
        self.model = nn.DataParallel(self.model).cuda()
        self.criterion = self.criterion.cuda()
        

    def update_hyperparam_epoch(self):
        init_lr = self.init_lr
        self.N = self.max_N
        
        if len(self.decay_epochs) > 0:
            lr = init_lr
            for decay in self.decay_epochs:
                if self.epoch >= decay: 
                    lr = lr * self.lr_decay
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.lr = lr

        
    def load_model(self, epoch=0):
        if epoch==0:
            file_name = self.get_file_name()
            checkpoint_dir = CHECKPOINT_ROOT / self.dataset / file_name
            checkpoint_path = max((f.stat().st_mtime, f) for f in checkpoint_dir.glob('*.pth'))[1]
            self.logger.info('Resume Latest from {}'.format(checkpoint_path))
        else:
            self.logger.info('Resume from {}'.format(epoch))
            file_name = self.get_file_name()
            checkpoint_path = CHECKPOINT_ROOT / self.dataset / file_name / 'e{:04d}.pth'.format(epoch)
            
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict((checkpoint['state_dict'])) # Set CUDA before if error occurs.
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']

    

    def save_checkpoint(self):
        save_path = self.get_save_path()
        save_file = 'e{:04d}.pth'.format(self.epoch+1)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
                'epoch': self.epoch,
                'arch': self.arch,
                'state_dict': self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                }, save_path / save_file )

        self.logger.info("Saved a checkpoint {}.".format(save_path / save_file))

        
    def set_dataset(self, dataset, test_splits=[], test_dataset=None):
        assert len(test_splits) > 0
        
        if not test_dataset:
            test_dataset = dataset
        
        self.dataset = dataset
        train_set, train_loader = factory.get_dataset(
            dataset, DATA_ROOT, self.max_N, self.batch_size, self.img_size, 
            self.max_skip
        )
        self.train_set = train_set
        self.train_loader = train_loader

        val_sets, val_loaders = [], []
        for split in test_splits:
            val_set, val_loader = factory.get_dataset_test(test_dataset, split, DATA_ROOT, self.test_batch_size, self.img_size)
            val_sets.append(val_set)
            val_loaders.append(val_loader)
        self.val_sets = val_sets
        self.val_loaders = val_loaders

    
    def base_scheme(self, frames, gt_masks, words, eval=False):
        if eval:
            B, T, _, W, H = frames.size()
            est_masks = torch.zeros_like(frames).sum(2)

            loss = 0.0
            prev_frame, prev_mask = None, None
            
            for t in range(0, T):
                mask_pred, logit =\
                    self.model(prev_frame, prev_mask, frames[:,t], words, eval=True)

                loss += torch.mean(self.criterion(logit, gt_masks[:,t].long()))

                prev_frame, prev_mask = frames[:, t], mask_pred[:, 1]
                est_masks[:,t] = mask_pred[:,1].detach()
                
            return est_masks, loss, 0, T
                
            
        N = frames.size(1)
        est_masks = torch.zeros_like(gt_masks)
        loss = 0.0

        prev_frame, prev_mask = None, None 

        for n in range(0, N):
            mask_pred, logit =\
                self.model(prev_frame, prev_mask, frames[:,n], words, eval=False)
            
            loss += torch.mean(self.criterion(logit, gt_masks[:,n].long()))

            prev_frame = frames[:, n]
            prev_mask = gt_masks[:, n]
            est_masks[:,n] = mask_pred[:,1].detach() # cut grad
                
        return est_masks, loss, 0, N
    
    
        
            
    def train(self):
        self.epoch += 1
        
        for self.epoch in range(self.epoch, self.max_epoch):
            
            self.update_hyperparam_epoch()
            self.logger.info('=========== EPOCH {} | LR {} | N {} =========='.format(self.epoch+1, self.lr, self.N))

            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':2.4f')
            IoUs = [AverageMeter('IoU_{}'.format(i), ':3.4f') for i in range(self.max_N)]
            mIoU = AverageMeter('mIoU', ':3.4f')

            end = time.time()

            for i, V in enumerate(tqdm(self.train_loader, dynamic_ncols=True)):
                
                data_time.update(time.time() - end)

                frames, gt_masks, words, _ = V  
                frames, gt_masks, words = ToCuda([frames, gt_masks, words])   
                
                est_masks, loss, N_start, N_end = self.scheme(frames, gt_masks, words, eval=False)
                
                losses.update(loss.item(), N_end - N_start)
                iou = [0]*self.N
                for n in range(N_start, N_end):
                    iou_n = IoU(est_masks[:,n], gt_masks[:,n]) 
                    iou[n] = iou_n
                miou = sum(iou[N_start:N_end])/(N_end - N_start)
                
                for n in range(N_start, N_end):
                    IoUs[n].update(iou[n])
                mIoU.update(miou)
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                batch_time.update(time.time() - end)
                end = time.time()
                
                if (i+1) % 100 == 0 or (i+1) == len(self.train_loader):

                    self.logger.info('{} | E [{:d}] | I [{:d}] | {} | {} | {} | {}'.format(
                        self.arch, self.epoch+1, i+1, losses, mIoU, data_time, batch_time))


            # save a checkpoint
            save_every = self.save_every
            if self.epoch > self.decay_epochs[0]:
                save_every = min(self.save_every, 5)
            
            if (self.epoch + 1) % save_every == 0:
                self.save_checkpoint()
                del loss, frames, gt_masks, words, V, est_masks
                self.evaluate()
                torch.cuda.empty_cache()

                
                
    def evaluate(self):
        
        save_path = self.get_save_path()
        for split, val_loader in zip(self.splits, self.val_loaders):
            split_name = '{}_{}'.format(self.dataset, split)
            
            eval_path = save_path / 'evaluation' / split_name
            if not eval_path.exists():
                eval_path.mkdir(parents=True, exist_ok=True)

            eval_json = Dict()
            eval_json.arch = self.arch
            eval_json.epoch = self.epoch
            eval_json.dataset = self.dataset

            self.logger.info('=========== EVALUATE MODEL {} EPOCH {}  >>>>  {}/{} =========='.format(self.arch, self.epoch+1, self.dataset, split))

            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':2.4f')
            
            J = AverageMeter('J', ':3.4f')
            F = AverageMeter('F', ':3.4f')
            
            end = time.time()
            
            precs_thres = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
            precs = np.zeros(len(precs_thres))
            num_samples = 0
            
            with torch.no_grad():
                for i, V in enumerate(tqdm(val_loader, dynamic_ncols=True)):
                    
                    data_time.update(time.time() - end)

                    frames, gt_masks, words, ref_ids, num_frames, metas = V 
                    T = num_frames.max().item()
                    
                    frames, gt_masks = frames[:, :T] , gt_masks[:, :T]
                    B, T, _, W, H = frames.size()
                    (frames, gt_masks), pad = pad_divide_by([frames, gt_masks], 16, (W, H))
                    frames, gt_masks, words = ToCuda([frames, gt_masks, words])

                    est_masks, loss, N_start, N_end = self.scheme(frames, gt_masks, words, eval=True)
                    losses.update(loss.item(), N_end - N_start)
                        
                    for b, n_frame in enumerate(num_frames):
                        j_score = 0.
                        f_score = 0.
                        ious = []
                        for t in range(n_frame):
                            iou_t = IoU(est_masks[b:b+1,t], gt_masks[b:b+1,t])
                            j_score += iou_t
                            ious.append(iou_t)
                            f_score += db_eval_boundary(est_masks[b:b+1,t], gt_masks[b:b+1,t])

                        j_score /= float(n_frame)
                        f_score /= float(n_frame)
                        iou_all = IoU(est_masks[b], gt_masks[b])

                        J.update(j_score)
                        F.update(f_score)

                        eval_json.j_score[ref_ids[b]] = j_score
                        eval_json.f_score[ref_ids[b]] = f_score
                        eval_json.ious[ref_ids[b]] = ious
                        
                        # Precision
                        precs += (j_score>precs_thres).astype(int)
                        num_samples += 1


                    batch_time.update(time.time() - end)
                    end = time.time()

                        
            precs /= num_samples
    
            eval_json.average_J = J.avg
            eval_json.average_F = F.avg

            eval_json.prec5 = precs[0]
            eval_json.prec6 = precs[1]
            eval_json.prec7 = precs[2]
            eval_json.prec8 = precs[3]
            eval_json.prec9 = precs[4]

            
            json.dump(eval_json, open(eval_path / 'e{:04d}.json'.format(self.epoch+1), 'w'))
            
            self.logger.warning('{} | E [{:d}] | {} | {} | {} | {} | {}'.format(
                self.arch, self.epoch+1, losses, J, F, data_time, batch_time,
            ))

            del V, frames

            
            
    