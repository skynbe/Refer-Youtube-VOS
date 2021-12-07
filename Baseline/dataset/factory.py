from .refer_datasets import REFER_YV_2019

from torch.utils import data
from pathlib import Path
import os
import torch, pdb


def get_dataset(dataset, DATA_ROOT, N, batch_size, img_size=(320, 320), max_skip=2, jitter=True, scale=1.0):
    
    if 'refer-yv-2019' in dataset:
        split_type = 'train_full'
        
        trainset = REFER_YV_2019(data_root=DATA_ROOT / 'youtube-vos-2019', split=split_type, N=N, size=img_size, max_skip=max_skip, jitter=jitter, scale=scale)
        dataLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    else:
        raise ValueError
        
        
    return trainset, dataLoader
        

    
def get_dataset_test(dataset, split, DATA_ROOT, batch_size, img_size=(320, 320)):

    if 'refer-yv-2019' in dataset:
        testset = REFER_YV_2019(data_root=DATA_ROOT / 'youtube-vos-2019', split=split, size=img_size, eval=True)
        dataLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    else:
        raise ValueError
    
    return testset, dataLoader

