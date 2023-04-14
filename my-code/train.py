# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/1 22:52
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
import torch
import argparse




def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--createData', action='store_true', default=False)
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--pretrain', default=None, type=str, metavar='PATH',
                        help='path to latest pretrain model (default: None)')

    args = parser.parse_args()

    return args


