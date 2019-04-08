# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 15:48:36 2019

@author: Or
"""

import torch
import itertools

def stack_indices(idx,n_nodes):
    sum_n = 0
    for i in range(len(n_nodes)-1):
        sum_n += n_nodes[i]
        idx[i+1] += sum_n
    return idx

def stack_list_indices(idx,n_nodes):
    sum_n = 0
    #out = []
    for i in range(len(n_nodes)-1):
        sum_n += n_nodes[i]
        for j in range(len(idx[i+1])):
            idx[i+1][j] += sum_n
    merged = list(itertools.chain(*idx))
    return merged

def max_graph_array(array,n_outs,masks):
    adds = 0
    count = 0
    idx = []
    for i in range(len(n_outs)):
        arr = array[count:(count+n_outs[i]),0].squeeze()
        min_arr = arr.min().item()
        arr = arr - min_arr + 1.0
        arr[masks[i].squeeze()] = 0.0
        idx.append(arr.argmax() + adds)
        adds += n_outs[i]
    out_idx = torch.LongTensor(idx)
    out_val = array[out_idx]
    return out_idx,out_val