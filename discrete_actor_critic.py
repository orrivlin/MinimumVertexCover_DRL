# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:20:39 2018

@author: orrivlin
"""
import torch 
import numpy as np
import dgl
import torch.nn.functional as F
from copy import deepcopy as dc
from Models import ACNet
from Utils import stack_indices, stack_list_indices, max_graph_array
from log_utils import mean_val, logger


class DiscreteActorCritic:
    def __init__(self, problem, cuda_flag):
        self.problem = problem
        ndim = self.problem.get_graph_dims()
        if cuda_flag:
            self.model = ACNet(ndim,264,1).cuda()
        else:
            self.model = ACNet(ndim,264,1)
        self.gamma = 0.98
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0003)
        self.batch_size = 32
        self.num_episodes = 1
        self.cuda = cuda_flag
        self.log = logger()
        self.log.add_log('tot_return')
        self.log.add_log('TD_error')
        self.log.add_log('entropy')
        
    def run_episode(self):
        sum_r = 0
        state, done = self.problem.reset()
        [idx1,idx2] = self.problem.get_ilegal_actions(state)
        t = 0
        while done == False:
            G = dc(state.g)
            if self.cuda:
                G.ndata['x'] = G.ndata['x'].cuda()
            [pi,val] = self.model(G)
            pi = pi.squeeze()
            pi[idx1] = -float('Inf')
            pi = F.softmax(pi,dim=0)
            dist = torch.distributions.categorical.Categorical(pi)
            action = dist.sample()
            
            new_state, reward, done = self.problem.step(dc(state),action)
            [idx1,idx2] = self.problem.get_ilegal_actions(new_state)
            state = dc(new_state)
            sum_r += reward
            if (t==0):
                PI = pi[action].unsqueeze(0)
                R = reward.unsqueeze(0)
                V = val.unsqueeze(0)
                t += 1
            else:
                PI = torch.cat([PI,pi[action].unsqueeze(0)],dim=0)
                R = torch.cat([R,reward.unsqueeze(0)],dim=0)
                V = torch.cat([V,val.unsqueeze(0)],dim=0)
        self.log.add_item('tot_return',sum_r.item())
        tot_return = R.sum().item()
        for i in range(R.shape[0] - 1):
            R[-2-i] = R[-2-i] + self.gamma*R[-1-i]
            
        return PI, R, V, tot_return
    
    
    def update_model(self,PI,R,V):
        self.optimizer.zero_grad()
        if self.cuda:
            R = R.cuda()
        A = R.squeeze() - V.squeeze().detach()
        L_policy = -(torch.log(PI)*A).mean()
        L_value = F.smooth_l1_loss(V.squeeze(), R.squeeze())
        L_entropy = -(PI*PI.log()).mean()
        L = L_policy + L_value - 0.1*L_entropy
        L.backward()
        self.optimizer.step()
        self.log.add_item('TD_error',L_value.detach().item())
        self.log.add_item('entropy',L_entropy.cpu().detach().item())
        
    
    def train(self):
        mean_return = 0
        for i in range(self.num_episodes):
            [pi,r,v,tot_return] = self.run_episode()
            mean_return = mean_return + tot_return
            if (i == 0):
                PI = pi
                R = r
                V = v
            else:
                PI = torch.cat([PI,pi],dim=0)
                R = torch.cat([R,r],dim=0)
                V = torch.cat([V,v],dim=0)
                
        mean_return = mean_return/self.num_episodes
        self.update_model(PI,R,V)
        return self.log
        