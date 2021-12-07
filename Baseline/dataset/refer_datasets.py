import os
import os.path as osp
import numpy as np
from PIL import Image

import collections
import torch
import torchvision
from torch.utils import data

import scipy.io
import glob, pdb
import time
import cv2
import random
import csv
import json
import pickle
from tqdm import tqdm
import torch.nn.functional as F

from itertools import chain, combinations

import tqdm

import matplotlib.pyplot as plt

def _flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


import pdb
from torch.utils import data
from pathlib import Path
from time import time

from utils.word_utils import Corpus


class REFER_YV_2019(data.Dataset):
    
    def __init__(self, data_root, split, N=2, size=(256,256), max_skip=1, query_len=20, eval=False, jitter=True, bert=False, scale=1.0):
        self.data_root = Path(data_root)
    
        split_type = split.split('_')[0]
    
        self.split = split_type
        
        self.N = N
        self.size = size
        self.max_skip = max_skip
        self.query_len = query_len
        self.eval = eval
        self.jitter = jitter
        self.bert = bert
        self.scale = scale
        
        self.set_corpus()
        self.max_frames = 36
        
        self.image_dir = self.data_root / split / 'JPEGImages'
        self.mask_dir = self.data_root / split / 'Annotations'
        
        self.set_meta_file()
        
        
    def set_meta_file(self):

        mymeta_path = self.data_root / self.split / 'mymeta.pkl'
        if mymeta_path.exists():
            with mymeta_path.open('rb') as f:
                self.videos = pickle.load(f)
        else:
            data = json.load(open(self.data_root / self.split / 'meta_expressions.json'))
            
            self.videos = []
            for vid, objs in tqdm.tqdm(data['videos'].items(), desc='Data processing'):
                    
                if self.eval:
                    for obj_id, obj in objs['objects'].items():
                        oid = int(obj_id)
                        
                        sents = obj['expressions']
                        if len(sents) > 0:
                            self.videos.append([vid, oid, obj['category'], obj['frames'], sents[0]])
                            
                    
                else: 
                    for obj_id, obj in objs['objects'].items():
                        oid = int(obj_id)
                            
                        sents = obj['expressions']
                        if len(sents) == 0:
                            print("Not included (no sents): ", vid, oid)

                        anker = []
                        for frm in obj['frames']:
                            mask_name = self.data_root / self.split / 'Annotations' / vid / '{}.png'.format(frm)
                            mask = np.uint8(Image.open(mask_name).convert('P'))
                            mask = np.uint8(mask == oid)
                            if float(mask.sum()) / mask.size > np.square(0.02):
                                anker += [1]
                            else:
                                anker += [0]

                        if sum(anker) >= 3:
                            for sent in sents:
                                self.videos.append([vid, oid, obj['category'], obj['frames'], anker, sent])
                        else:
                            print("Not included : ", vid, oid, anker)
            
            
            with mymeta_path.open('wb') as f:
                pickle.dump(self.videos, f, pickle.HIGHEST_PROTOCOL)
                
        len_videos = len(self.videos)
        if self.scale < 1.0:
            len_videos = int(len_videos * self.scale)
        self.videos = self.videos[:len_videos]


    def __len__(self):
        len_videos = len(self.videos)
        return len_videos

    
    def set_corpus(self):
        self.corpus = Corpus()
        #TODO: ref file
        vocab_path = self.data_root / 'vocabulary_Gref.txt'
        corpus_path = self.data_root / 'corpus.pth'
        if not corpus_path.exists():
            print('Saving dataset corpus dictionary...')
            self.corpus.load_file(vocab_path)
            torch.save(self.corpus, corpus_path)
        else:
            self.corpus = torch.load(corpus_path)

            
    def random_crop(self, frame, mask, size, rnd):

        # resize `frame` before cropping
        # resized frame should be large than `size` but shouldn't be too large
        min_scale = np.maximum(size[0]/np.float(frame.shape[0]), size[1]/np.float(frame.shape[1]))
        scale = np.maximum(rnd.uniform(min_scale+0.01, 1.875*min_scale), min_scale+0.01)

        dsize = (np.int(frame.shape[1]*scale), np.int(frame.shape[0]*scale))
        trans_frame  = cv2.resize(frame, dsize=dsize, interpolation=cv2.INTER_LINEAR)
        trans_mask = cv2.resize(mask, dsize=dsize, interpolation=cv2.INTER_NEAREST)
        
        ## try to crop patch that contains object area if possible, otherwise just return
        np_in1 = np.sum(trans_mask)

        for _ in range(100):
            cr_y = rnd.randint(0, trans_mask.shape[0] - size[0])
            cr_x = rnd.randint(0, trans_mask.shape[1] - size[1])
            crop_mask = trans_mask[cr_y:cr_y+size[0], cr_x:cr_x+size[1]]
            crop_frame = trans_frame[cr_y:cr_y+size[0], cr_x:cr_x+size[1],:]

            nnz_crop_mask = np.sum(crop_mask)
            break

        return crop_frame, crop_mask
    
    
    def random_jitter(self, frame, mask, size, rnd):

        scale = rnd.uniform(1, 1.1)
        dsize = (int(size[0]*scale), int(size[1]*scale))

        trans_frame  = cv2.resize(frame, dsize=dsize, interpolation=cv2.INTER_LINEAR)
        trans_mask = cv2.resize(mask, dsize=dsize, interpolation=cv2.INTER_NEAREST)
        
        np_in1 = np.sum(trans_mask)

        crop_frame = None
        for _ in range(100):
            cr_y = rnd.randint(0, trans_mask.shape[0] - size[0])
            cr_x = rnd.randint(0, trans_mask.shape[1] - size[1])
            crop_mask = trans_mask[cr_y:cr_y+size[0], cr_x:cr_x+size[1]]
            crop_frame = trans_frame[cr_y:cr_y+size[0], cr_x:cr_x+size[1],:]
            if np.sum(crop_mask) > 0.8*np_in1:
                break
                
        if crop_frame is None:
            return self.random_jitter(frame, mask, size, rnd)

        return crop_frame, crop_mask

    
    def resize(self, frame, mask, size):
        scale = np.maximum(size[0]/np.float(frame.shape[0]), size[1]/np.float(frame.shape[1]))
        dsize = (np.int(frame.shape[1]*scale), np.int(frame.shape[0]*scale))
        size = (size[0], size[1])
        resize_frame  = cv2.resize(frame, dsize=size, interpolation=cv2.INTER_LINEAR)
        resize_mask = cv2.resize(mask, dsize=size, interpolation=cv2.INTER_NEAREST)
        return resize_frame, resize_mask
    
    
    def load_pair(self, vid, oid, fid):
        img_name = self.data_root / self.split / 'JPEGImages' / vid / '{}.jpg'.format(fid)
        mask_name = self.data_root / self.split / 'Annotations' / vid / '{}.png'.format(fid)
    
        frame = np.float32(Image.open(img_name).convert('RGB')) / 255.
        mask = np.uint8(Image.open(mask_name).convert('P'))
        mask = np.uint8(mask == oid)
        return frame, mask
    
    
    def load_pairs(self, vid, oid, frame_ids):
        frames, masks = [], []
        for frame_id in frame_ids:
            frame, mask = self.load_pair(vid, oid, frame_id)
            frame, mask = self.resize(frame, mask, self.size)
            frames.append(frame)
            masks.append(mask)
            
        N_frames = np.stack(frames, axis=0)
        N_masks = np.stack(masks, axis=0)
        
        Fs = torch.from_numpy(np.transpose(N_frames, (0, 3, 1, 2)).copy()).float()
        Ms = torch.from_numpy(N_masks.copy()).float()
        return Fs, Ms
    
    
    def sample_frame_ids_base(self, frame_ids, anker, rnd, vid, oid):
        
        n_images = len(frame_ids)
        tt = 0
        while True:
            sample_skips = [rnd.randint(1, self.max_skip) for _ in range(1, self.N)]
            if sum(sample_skips) < n_images:
                break
            if tt > 100:
                sample_skips = [1] * (self.N-1)
                break
            tt = tt+1
        
        use_ids = [None] * self.N
        # start index
        anker_nnz = [i for i, e in enumerate(anker) if e != 0]
        n_skip = sum(sample_skips)

        if True: # always forward
            anker_idx = [ i for i in anker_nnz if i + n_skip < n_images ]
            if len(anker_idx) > 0:
                use_ids[0] = anker_idx[ rnd.randint(0, len(anker_idx)-1) ]
            else:
                use_ids[0] = anker_nnz[0]

        for i in range(1, self.N):
            use_ids[i] = (use_ids[i-1] + sample_skips[i-1]) % n_images
            
        use_frame_ids = [frame_ids[i] for i in use_ids]
        return use_frame_ids
        
        

    def __getitem__(self, index):
        
        if self.eval:
            vid, oid, category, frame_ids, sent = self.videos[index]
            
            Fs, Ms = self.load_pairs(vid, oid, frame_ids)
            
            num_frames = len(Fs)
            if num_frames < self.max_frames:
                pad_frames = self.max_frames-num_frames
                Fs = F.pad(Fs, (0,0)*3+(0,pad_frames))
                Ms = F.pad(Ms, (0,0)*2+(0,pad_frames))

            words = self.tokenize_sent(sent)
            ann_id = '{}_{}'.format(vid, oid)
            
            meta = {'sent': sent}
            return Fs, Ms, words, ann_id, num_frames, meta 
            
            
        rnd = random.Random()

        vid, oid, category, frame_ids, anker, sent = self.videos[index]
        use_frame_ids = self.sample_frame_ids_base(frame_ids, anker, rnd, vid, oid)
        
        # get frames and masks
        frames, masks = [], []
        for frame_id in use_frame_ids:
            frm, msk = self.load_pair(vid, oid, frame_id)
            if self.jitter:
                frm, msk = self.random_jitter(frm, msk, self.size, rnd)
            else:
                frm, msk = self.resize(frm, msk, self.size)
            frames.append(frm)
            masks.append(msk)
            

        frames = np.stack(frames, axis=0)
        masks = np.stack(masks, axis=0)

        Fs = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2)).copy()).float()
        Ms = torch.from_numpy(masks.copy()).float()
        words = self.tokenize_sent(sent)
        ann_id = '{}_{}'.format(vid, oid)
        
        return Fs, Ms, words, ann_id
    
    
    def tokenize_sent(self, sent):
        return self.corpus.tokenize(sent, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.untokenize(words)

    
    