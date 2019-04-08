# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 07:57:52 2019

@author: Or
"""

import torch
from MVC import MVC
import dgl
import torch.nn.functional as F
from Models import ACNet
import time
from copy import deepcopy as dc
import matplotlib.pyplot as plt
import networkx as nx



cuda_flag = True
num_nodes = 40
p_edge = 0.15
mvc = MVC(num_nodes,p_edge)
ndim = mvc.get_graph_dims()

if cuda_flag:
    NN = ACNet(ndim,264,1).cuda()
else:
    NN = ACNet(ndim,264,1)
PATH = 'mvc_net.pt'
NN.load_state_dict(torch.load(PATH))

init_state,done = mvc.reset()
pos = nx.spring_layout(init_state.g.to_networkx(), iterations=20)

#### GCN Policy
state = dc(init_state)
if cuda_flag:
    state.g.ndata['x'] = state.g.ndata['x'].cuda()
sum_r = 0
T1 = time.time()
[idx1,idx2] = mvc.get_ilegal_actions(state)
while done == False:
    G = state.g
    [pi,val] = NN(G)
    pi = pi.squeeze()
    pi[idx1] = -float('Inf')
    pi = F.softmax(pi,dim=0)
    dist = torch.distributions.categorical.Categorical(pi)
    action = dist.sample()            
    new_state, reward, done = mvc.step(state,action)
    [idx1,idx2] = mvc.get_ilegal_actions(new_state)
    state = new_state
    sum_r += reward
T2 = time.time()

node_tag = state.g.ndata['x'][:,0].cpu().squeeze().numpy().tolist()
nx.draw(state.g.to_networkx(), pos, node_color=node_tag, with_labels=True)
plt.show()



### Heuristic Policy
state = dc(init_state)
done = False
sum_r2 = 0
T1 = time.time()
[idx1,idx2] = mvc.get_ilegal_actions(state)
while done == False:
    G = state.g
    un_allowed = idx1.numpy().squeeze()
    degree = G.in_degrees() + G.out_degrees()
    degree[un_allowed] = 0
    degree = torch.Tensor(np.array(degree))
    action = degree.argmax()
    
    new_state, reward, done = mvc.step(state,action)
    [idx1,idx2] = mvc.get_ilegal_actions(new_state)
    state = new_state
    sum_r2 += reward
T2 = time.time()

node_tag = state.g.ndata['x'][:,0].cpu().squeeze().numpy().tolist()
nx.draw(state.g.to_networkx(), pos, node_color=node_tag, with_labels=True)
plt.show()

print('Ratio: {}'.format((sum_r/sum_r2).item()))
