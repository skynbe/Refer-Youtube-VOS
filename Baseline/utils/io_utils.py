import os
import os.path
import hashlib
import errno
import logging
from collections import defaultdict
from string import Formatter
import torch
import re, pdb
from datetime import datetime

import cv2
import numpy as np
from utils.helpers import *
import pdb
from PIL import Image
from matplotlib import pyplot as plt




def get_logger(name, fmt='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
               print_level=logging.INFO,
               write_level=logging.DEBUG, log_file='', mode='w'):
    """
    Get Logger with given name
    :param name: logger name.
    :param fmt: log format. (default: %(asctime)s:%(levelname)s:%(name)s:%(message)s)
    :param level: logging level. (default: logging.DEBUG)
    :param log_file: path of log file. (default: None)
    :return:
    """
    logger = logging.getLogger(name)
    #  logger.setLevel(write_level)
    logging.basicConfig(level=print_level)
    formatter = logging.Formatter(fmt, datefmt='%Y/%m/%d %H:%M:%S')

    # Add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(write_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    if print_level is not None:
        try:
            import coloredlogs
            coloredlogs.install(level=print_level, logger=logger)
            coloredlogs.DEFAULT_LEVEL_STYLES = {'critical': {'color': 'red', 'bold': True}, 'debug': {'color': 'green'}, 'error': {'color': 'red'}, 'info': {}, 'notice': {'color': 'magenta'}, 'spam': {'color': 'green', 'faint': True}, 'success': {'color': 'green', 'bold': True}, 'verbose': {'color': 'blue'}, 'warning': {'color': 'yellow'}}
        except ImportError:
            print("Please install Coloredlogs for better view")
            # Add stream handler
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(print_level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
    return logger




def print_table(JF):
    from prettytable import PrettyTable
    R = JF.shape[1]
    JFmean = np.mean(JF, axis=0)
    Jmeans, Fmeans = [], []
    for r in range(R):
        Jmeans.append('{: .3f}'.format(JFmean[r,0]))
        Fmeans.append('{: .3f}'.format(JFmean[r,1]))

    pt = PrettyTable()
    pt.field_names = ['Round', '1', '2', '3', '4', '5']
    pt.add_row(["J mean"] + Jmeans)
    pt.add_row(["F mean"] + Fmeans)
    return pt

            
def save_result(path, tf, te, tm, size=[360,640], n=0): 
    N = tf.size()[1]                # b, c, f, h, w  // b, f, h, w // b, 4, f
    tf = (tf.data[n].permute(0,2,3,1) * 255.).cpu().numpy().astype(np.uint8)
    te = (te.data[n].cpu().numpy() > 0.5).astype(np.int)
    tm = (tm.data[n].cpu().numpy() > 0.5).astype(np.int)
    
    # canvas = np.zeros(((N+1)*size[0], size[1], 3), dtype=np.uint8)
    canvas = np.zeros((2*size[0], (N)*size[1], 3), dtype=np.uint8)

    for i in range(N):
        ov_g = overlay_mask(tf[i], tm[i])
        ov_e = overlay_mask(tf[i], te[i])
        canvas[0:size[0],(i)*size[1]:(i+1)*size[1],:] = ov_g
        canvas[size[0]:2*size[0],(i)*size[1]:(i+1)*size[1],:] = ov_e

    im = Image.fromarray(canvas)
    im.save(path)