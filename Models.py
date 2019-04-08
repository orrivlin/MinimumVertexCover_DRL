# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:28:50 2019

@author: orrivlin
"""

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F



msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    accum = torch.cat((torch.mean(nodes.mailbox['m'], 1),torch.max(nodes.mailbox['m'],1)[0]),dim=1)
    return {'hm': accum}

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(3*in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(torch.cat((node.data['h'],node.data['hm']),dim=1))
        h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        g.ndata.pop('hm')
        return g.ndata.pop('h')


class ACNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(ACNet, self).__init__()
        
        self.policy = nn.Linear(hidden_dim,1)
        self.value = nn.Linear(hidden_dim,1)
        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])

    def forward(self, g):
        h = g.ndata['x']
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        mN = dgl.mean_nodes(g, 'h')
        PI = self.policy(g.ndata['h'])
        V = self.value(mN)
        g.ndata.pop('h')
        return PI, V
    