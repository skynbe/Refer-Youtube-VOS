# -*- coding: utf-8 -*-
from __future__ import division
import warnings
warnings.simplefilter("ignore", UserWarning)

import argparse
from trainer import Trainer

import os, torch

def main():
    def get_arguments():
        parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
        parser.add_argument("--arch", type=str, default='', help='base_model')

        parser.add_argument("--desc", type=str, default='')
        parser.add_argument("--eval", action='store_true')
        parser.add_argument("--eval_first", action='store_true')

        parser.add_argument("--init_lr", type=float, default=1e-4, help="init lr")
        parser.add_argument("--batch_size", type=int, default=16, help="batch size")
        parser.add_argument("--test_batch_size", type=int, default=0, help="batch size for eval")
        
        parser.add_argument("--img_size", type=int, default=320, help="image size")
        parser.add_argument("--max_epoch", type=int, default=150, help="lr decay epoch")
        parser.add_argument("--decay_epochs", type=int, default=[], nargs='+', help='50 80 100')  
        
        parser.add_argument("--optimizer", type=str, default='adam')
        parser.add_argument("--lr_decay", type=float, default=0.1, help="lr decay")
        parser.add_argument("--save_every", type=int, default=0, help="save_every")
        
        parser.add_argument("--max_N", type=int, default=0, help="lr decay epoch")
        parser.add_argument("--max_skip", type=int, default=2, help="max_skip for video")
        
        parser.add_argument("--dataset", type=str, default='')  
        parser.add_argument("--test_dataset", type=str, default=None)  
        parser.add_argument("--splits", type=str, default=[], nargs='+', help='test splits')  

        parser.add_argument("--checkpoint", type=str, default='')  
        parser.add_argument("--epoch", type=int, default=-1, help="resume training from the specified epoch number.")
        
        return parser.parse_args()

    args = get_arguments()
    
    trainer = Trainer(args)
    trainer.cuda()
    trainer.set_dataset(args.dataset, args.splits, args.test_dataset)
    if args.epoch != -1 or args.eval:
        trainer.load_model(args.epoch)

    trainer.model.eval() # turn-off BN
    if args.eval or args.eval_first:
        trainer.evaluate()
        if not args.eval_first:
            return
    trainer.train()
    
if __name__ == "__main__":
    main()
