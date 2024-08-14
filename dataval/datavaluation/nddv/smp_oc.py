#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2024/02/19 09:31:03
@Author  :   Liang Zhangyong
@Contact :   liangzhangyong1994@gmail.com
'''

# here put the import lib
import numpy as np
from collections import OrderedDict
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.autograd as autograd
from scipy.stats import multivariate_normal as normal

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
    

class Dense(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        activate: bool = True,
    ):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activate = activate
        self.linear = nn.Linear(input_dim, output_dim)
        # self.act = nn.ReLU()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activate:
            return self.act(self.linear(x))
        else:
            return self.linear(x)
        
        
class ControlNN(nn.Module):
    def __init__(self, num_hiddens: list[int]):
        super(ControlNN, self).__init__()
        self.bn = nn.BatchNorm1d(num_hiddens[0])
        self.layers = [Dense(num_hiddens[i-1],num_hiddens[i]) for i in range(1, len(num_hiddens)-1)]
        self.layers += [Dense(num_hiddens[-2], num_hiddens[-1],activate=False)]
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.bn(x))
    
    
class ClassifierSMP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        batch_size: int,
        hidden_dim: int = 25,
        Ntime:int = 2,
        totalT:int = 1,
        sigma:float = 0.01,
        interact:float = 1,
        act_fn: Optional[Callable] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super(ClassifierSMP, self).__init__()
        self.device = device
        act_fn = nn.ReLU() if act_fn is None else act_fn
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.Ntime = Ntime
        self.totalT = totalT
        self.sigma = sigma
        self.interact = interact
        self.num_classes = num_classes
        self.dt = self.totalT/self.Ntime
        # self.dw=torch.FloatTensor(normal.rvs(size=[self.batch_size,self.input_dim])*np.sqrt(self.dt))
        self.dw = torch.randn(self.batch_size,self.input_dim, device=self.device)*np.sqrt(self.dt)
        self.hidden = [input_dim, hidden_dim, hidden_dim, input_dim]
        self.mfg = nn.ModuleList([ControlNN(self.hidden) for _ in range(self.Ntime)])
        self.linear = nn.Linear(input_dim, num_classes)
        self.softmax = nn.Softmax(-1)
        # self.apply(_weights_init)
        
    def forward(self, bf):
        if bf.mode == 'forwardX':
            if bf.mf:
                return self.forwardX(bf.inputs, bf.mu)
            else:
                return self.forwardX(bf.inputs)
        elif bf.mode == 'HamGraFlow':
            return self.HamGraFlow(bf.xMat, bf.wMat, bf.loss)
        elif bf.mode == 'backwardYZ':
            if bf.mf:
                return self.backwardYZ(bf.xMat, bf.wMat, bf.loss, bf.mu)
            else:
                return self.backwardYZ(bf.xMat, bf.wMat, bf.loss)
        elif bf.mode == 'HamCompute':
            if bf.mf:
                return self.HamCompute(bf.xMat, bf.yMat, bf.zMat, bf.mu)
            else:
                return self.HamCompute(bf.xMat, bf.yMat, bf.zMat)
        elif bf.mode == 'ValueEstimator':
            return self.ValueEstimator(bf.xMat, bf.loss)
    
    # Define the forward process: Euler-Maruyama method
    def forwardX(self, x, mu=None):
        xMat = []
        wMat = []
        x0=torch.clone(x)
        xMat.append(x0)
        
        if x0.shape[0] == self.batch_size:
            dw = self.dw
        else:
            # dw=torch.FloatTensor(normal.rvs(size=x0.shape)*np.sqrt(self.dt))
            dw=torch.randn(x0.shape, device=self.device)*np.sqrt(self.dt)
            
        for i in range(self.Ntime):
            # x_mu = torch.cat((x0, mu), dim=1)
            # x0 = x0 + self.mfg_mf[i](x_mu)*self.dt + self.sigma*dw
            control = self.mfg[i](x0)
            if mu is not None:
                x0 = x0 + (self.interact*(mu - x0) + control)*self.dt + self.sigma*dw
            else:
                x0 = x0 + control*self.dt + self.sigma*dw
            xMat.append(x0)
            wMat.append(dw)
            
        # state of the batch data points
        state = x0
            
        x0=self.linear(x0)
        xMat.append(x0)
        
        x0=self.softmax(x0)
        xMat.append(x0)
        
        return xMat, wMat, state
    
    # Define the backward process: HJB equation to compute the optimal control loss
    def backwardYZ(self, xMat, wMat, loss_val, mu=None): 
        yMat=[]
        zMat=[]
        L=len(xMat)
        x_terminal=xMat[-1]     # The terminal X_T
        
        # This very first step is required, as it is the fully-connected layer. 
        y_terminal = autograd.grad(outputs=[loss_val], inputs=[x_terminal],
                                   grad_outputs=torch.ones_like(loss_val), allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(y_terminal)  # The terminal YT
        
        # Here y_terminal has dim batch_size x output_size (2 x 2)
        x_pro=xMat[L-2]
        hami_pro = torch.sum(y_terminal.detach()*self.softmax(x_pro), dim=1, keepdim=True)
        hami_pro = hami_pro.view(-1,1)
        hami_pro_x = -autograd.grad(outputs=[hami_pro], inputs=[x_pro],
                               grad_outputs=torch.ones_like(hami_pro),allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(hami_pro_x)
        
        # The last block is stand-alone FC, must be treated separately. 
        x_linear=xMat[L-3]
        hami=torch.sum(hami_pro_x.detach()*self.linear(x_linear), dim=1, keepdim=True) 
        hami=hami.view(-1,1)
        hami_x = -autograd.grad(outputs=[hami], inputs=[x_linear],
                             grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(hami_x)
        
        # finding the optimal control
        for i in range(self.Ntime-1, -1, -1):
            zMat.append(yMat[-1]*wMat[i]/self.dt)
            X=xMat[i]
            if i == 0:
                X.requires_grad = True
            control = self.mfg[i](X)
            if mu is not None:
                hami = torch.sum(yMat[-1].detach()*(self.interact*(mu.detach()-X.detach()) + control) + self.sigma*zMat[-1].detach(), dim=1, keepdim=True)
            else:
                hami = torch.sum(yMat[-1].detach()*control + self.sigma*zMat[-1].detach(), dim=1, keepdim=True)
            hami = hami.view(-1,1)
            hami_x = autograd.grad(outputs=[hami], inputs=[X],
                                   grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
            ytemp = yMat[-1] - hami_x*self.dt + zMat[-1]*wMat[i]
            yMat.append(ytemp)
                
        return yMat, zMat
    
    # Compute the Hamiltonian     H = P*F + σ*Z - L
    def HamCompute(self, xMat, yMat, zMat, mu=None):
        totalham=0.0
        L=len(xMat)
        hami_pro = torch.mean(torch.sum(yMat[0].detach()*self.softmax(xMat[L-2]), dim=1, keepdim=True))
        totalham += hami_pro
        hami_linear = torch.mean(torch.sum(yMat[1].detach()*self.linear(xMat[L-3]), dim=1, keepdim=True))
        totalham += hami_linear
        
        for i in range(self.Ntime):
            control = self.mfg[i](xMat[i].detach())
            if mu is not None:
                hami = torch.mean(torch.sum(yMat[len(self.mfg)+2-i-1].detach()*(self.interact*(mu.detach()-xMat[i].detach())+control) + self.sigma*zMat[i].detach(), dim=1, keepdim=True))
            else:
                hami = torch.mean(torch.sum(yMat[len(self.mfg)+2-i-1].detach()*control + self.sigma*zMat[i].detach(), dim=1, keepdim=True))
            totalham += hami
            
        return totalham/self.batch_size/self.Ntime
    
    def ValueEstimator(self, xMat, loss_val): 
        yMat=[]
        L=len(xMat)
        x_terminal=xMat[-1]     # The terminal X_T
        
        # This very first step is required, as it is the fully-connected layer. 
        y_terminal = autograd.grad(outputs=[loss_val], inputs=[x_terminal],
                                   grad_outputs=torch.ones_like(loss_val), allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(y_terminal)  # The terminal YT
        
        # Here y_terminal has dim batch_size x output_size (2 x 2)
        x_pro=xMat[L-2]
        hami_pro = torch.sum(y_terminal.detach()*self.softmax(x_pro), dim=1, keepdim=True)
        hami_pro = hami_pro.view(-1,1)
        hami_pro_x = -autograd.grad(outputs=[hami_pro], inputs=[x_pro],
                               grad_outputs=torch.ones_like(hami_pro),allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(hami_pro_x)
        
        # The last block is stand-alone FC, must be treated separately. 
        x_linear=xMat[L-3]
        hami=torch.sum(hami_pro_x.detach()*self.linear(x_linear), dim=1, keepdim=True) 
        hami=hami.view(-1,1)
        data_values = -autograd.grad(outputs=[hami], inputs=[x_linear],
                             grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
            
        data_values = torch.mean(data_values, dim=1, keepdim=True)
                
        return data_values
    
    def HamGraFlow(self, xMat, wMat, loss_val):
        yMat=[]
        zMat=[]
        L=len(xMat)
        x_terminal=xMat[-1]     # The terminal X_T
        
        # This very first step is required, as it is the fully-connected layer. 
        y_terminal = autograd.grad(outputs=[loss_val], inputs=[x_terminal],
                                   grad_outputs=torch.ones_like(loss_val), allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(y_terminal)  # The terminal YT
        
        # Here y_terminal has dim batch_size x output_size (2 x 2)
        x_pro=xMat[L-2]
        hami_pro = torch.sum(y_terminal.detach()*self.softmax(x_pro), dim=1, keepdim=True)
        hami_pro = hami_pro.view(-1,1)
        hami_pro_x = -autograd.grad(outputs=[hami_pro], inputs=[x_pro],
                               grad_outputs=torch.ones_like(hami_pro),allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(hami_pro_x)
        
        # The last block is stand-alone FC, must be treated separately. 
        x_linear=xMat[L-3]
        hami=torch.sum(hami_pro_x.detach()*self.linear(x_linear), dim=1, keepdim=True) 
        hami=hami.view(-1,1)
        hami_x = -autograd.grad(outputs=[hami], inputs=[x_linear],
                             grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(hami_x)
        
        # finding the optimal control
        for i in range(self.Ntime-1, -1, -1):
            zMat.append(yMat[-1]*wMat[i]/self.dt)
            X=xMat[i]
            if i == 0:
                X.requires_grad = True
            control = self.mfg[i](X)
            hami = torch.sum(yMat[-1].detach()*control + self.sigma*zMat[-1].detach(), dim=1, keepdim=True)
            hami = hami.view(-1,1)
            hami_x = -autograd.grad(outputs=[hami], inputs=[X],
                                   grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
        return hami_x      
    
    
class RegressionSMP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        batch_size: int,
        hidden_dim: int = 25,
        Ntime:int = 2,
        totalT:int = 1,
        sigma:float = 0.01,
        interact:float = 1,
        act_fn: Optional[Callable] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super(RegressionSMP, self).__init__()
        self.device = device
        act_fn = nn.ReLU() if act_fn is None else act_fn
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.Ntime = Ntime
        self.totalT = totalT
        self.sigma = sigma
        self.interact = interact
        self.num_classes = num_classes
        self.dt = self.totalT/self.Ntime
        self.dw = torch.randn(self.batch_size,self.input_dim, device=self.device)*np.sqrt(self.dt)
        # self.dw=torch.FloatTensor(normal.rvs(size=[self.batch_size,self.input_dim])*np.sqrt(self.dt))
        self.hidden = [input_dim, hidden_dim, hidden_dim, input_dim]
        self.mfg = nn.ModuleList([ControlNN(self.hidden) for _ in range(self.Ntime)])
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, bf):
        if bf.mode == 'forwardX':
            if bf.mf:
                return self.forwardX(bf.inputs, bf.mu)
            else:
                return self.forwardX(bf.inputs)
        elif bf.mode == 'HamGraFlow':
            return self.HamGraFlow(bf.xMat, bf.wMat, bf.loss)
        elif bf.mode == 'backwardYZ':
            if bf.mf:
                return self.backwardYZ(bf.xMat, bf.wMat, bf.loss, bf.mu)
            else:
                return self.backwardYZ(bf.xMat, bf.wMat, bf.loss)
        elif bf.mode == 'HamCompute':
            if bf.mf:
                return self.HamCompute(bf.xMat, bf.yMat, bf.zMat, bf.mu)
            else:
                return self.HamCompute(bf.xMat, bf.yMat, bf.zMat)
        elif bf.mode == 'ValueEstimator':
            return self.ValueEstimator(bf.xMat, bf.loss)
    
    # Define the forward process: Euler-Maruyama method
    def forwardX(self, x, mu=None):
        xMat = []
        wMat = []
        x0=torch.clone(x)
        xMat.append(x0)
        
        if x0.shape[0] == self.batch_size:
            dw = self.dw
        else:
            # dw=torch.FloatTensor(normal.rvs(size=x0.shape)*np.sqrt(self.dt))
            dw=torch.randn(x0.shape, device=self.device)*np.sqrt(self.dt)
            
        for i in range(self.Ntime):
            # x_mu = torch.cat((x0, mu), dim=1)
            # x0 = x0 + self.mfg_mf[i](x_mu)*self.dt + self.sigma*dw
            control = self.mfg[i](x0)
            if mu is not None:
                x0 = x0 + (self.interact*(mu - x0) + control)*self.dt + self.sigma*dw
            else:
                x0 = x0 + control*self.dt + self.sigma*dw
            xMat.append(x0)
            wMat.append(dw)
            
        # state of the batch data points
        state = x0
            
        x0=self.linear(x0)
        xMat.append(x0)
        
        return xMat, wMat, state
    
    # Define the backward process: HJB equation to compute the optimal control loss
    def backwardYZ(self, xMat, wMat, loss_val, mu=None): 
        yMat=[]
        zMat=[]
        L=len(xMat)
        x_terminal=xMat[-1]     # The terminal X_T
        
        # This very first step is required, as it is the fully-connected layer. 
        y_terminal = autograd.grad(outputs=[loss_val], inputs=[x_terminal],
                                   grad_outputs=torch.ones_like(loss_val), allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(y_terminal)  # The terminal YT
        
        # The last block is stand-alone FC, must be treated separately. 
        x_linear=xMat[L-2]
        hami=torch.sum(y_terminal.detach()*self.linear(x_linear), dim=1, keepdim=True) 
        hami=hami.view(-1,1)
        hami_x = -autograd.grad(outputs=[hami], inputs=[x_linear],
                             grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(hami_x)
        
        # finding the optimal control
        for i in range(self.Ntime-1, -1, -1):
            zMat.append(yMat[-1]*wMat[i]/self.dt)
            X=xMat[i]
            if i == 0:
                X.requires_grad = True
            control = self.mfg[i](X)
            if mu is not None:
                hami = torch.sum(yMat[-1].detach()*(self.interact*(mu.detach()-X.detach()) + control) + self.sigma*zMat[-1].detach(), dim=1, keepdim=True)
            else:
                hami = torch.sum(yMat[-1].detach()*control + self.sigma*zMat[-1].detach(), dim=1, keepdim=True)
            hami = hami.view(-1,1)
            hami_x = autograd.grad(outputs=[hami], inputs=[X],
                                   grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
            ytemp = yMat[-1] - hami_x*self.dt + zMat[-1]*wMat[i]
            yMat.append(ytemp)
                
        return yMat, zMat
    
    # Compute the Hamiltonian     H = P*F + σ*Z - L
    def HamCompute(self, xMat, yMat, zMat, mu=None):
        totalham=0.0
        L=len(xMat)
        hami_linear = torch.mean(torch.sum(yMat[0].detach()*self.linear(xMat[L-3]), dim=1, keepdim=True))
        totalham += hami_linear
        
        for i in range(self.Ntime):
            control = self.mfg[i](xMat[i].detach())
            if mu is not None:
                hami = torch.mean(torch.sum(yMat[len(self.mfg)+2-i-1].detach()*(self.interact*(mu.detach()-xMat[i].detach())+control) + self.sigma*zMat[i].detach(), dim=1, keepdim=True))
            else:
                hami = torch.mean(torch.sum(yMat[len(self.mfg)+2-i-1].detach()*control + self.sigma*zMat[i].detach(), dim=1, keepdim=True))
            totalham += hami
            
        return totalham/self.batch_size/self.Ntime
    
    def ValueEstimator(self, xMat, loss_val): 
        yMat=[]
        L=len(xMat)
        x_terminal=xMat[-1]     # The terminal X_T
        
        # This very first step is required, as it is the fully-connected layer. 
        y_terminal = autograd.grad(outputs=[loss_val], inputs=[x_terminal],
                                   grad_outputs=torch.ones_like(loss_val), allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(y_terminal)  # The terminal YT
        
        # The last block is stand-alone FC, must be treated separately. 
        x_linear=xMat[L-3]
        hami=torch.sum(y_terminal.detach()*self.linear(x_linear), dim=1, keepdim=True) 
        hami=hami.view(-1,1)
        hami_x = -autograd.grad(outputs=[hami], inputs=[x_linear],
                             grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(hami_x)
            
        value = torch.mean(yMat[-1], dim=1, keepdim=True)
                
        return value
    
    def HamGraFlow(self, xMat, wMat, loss_val):
        yMat=[]
        zMat=[]
        L=len(xMat)
        x_terminal=xMat[-1]     # The terminal X_T
        
        # This very first step is required, as it is the fully-connected layer. 
        y_terminal = autograd.grad(outputs=[loss_val], inputs=[x_terminal],
                                   grad_outputs=torch.ones_like(loss_val), allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(y_terminal)  # The terminal YT
        
        # The last block is stand-alone FC, must be treated separately. 
        x_linear=xMat[L-2]
        hami=torch.sum(y_terminal.detach()*self.linear(x_linear), dim=1, keepdim=True) 
        hami=hami.view(-1,1)
        hami_x = -autograd.grad(outputs=[hami], inputs=[x_linear],
                             grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(hami_x)
        
        # finding the optimal control
        for i in range(self.Ntime-1, -1, -1):
            zMat.append(yMat[-1]*wMat[i]/self.dt)
            X=xMat[i]
            if i == 0:
                X.requires_grad = True
            control = self.mfg[i](X)
            hami = torch.sum(yMat[-1].detach()*control + self.sigma*zMat[-1].detach(), dim=1, keepdim=True)
            hami = hami.view(-1,1)
            hami_x = -autograd.grad(outputs=[hami], inputs=[X],
                                   grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
        return hami_x
    
    
class LogisticRegressionSMP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        batch_size: int,
        hidden_dim: int = 25,
        Ntime:int = 2,
        totalT:int = 1,
        sigma:float = 0.01,
        interact:float = 1,
        act_fn: Optional[Callable] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super(LogisticRegressionSMP, self).__init__()
        self.device = device
        act_fn = nn.ReLU() if act_fn is None else act_fn
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.Ntime = Ntime
        self.totalT = totalT
        self.sigma = sigma
        self.interact = interact
        self.num_classes = num_classes
        self.dt = self.totalT/self.Ntime
        self.dw = torch.randn(self.batch_size,self.input_dim, device=self.device)*np.sqrt(self.dt)
        # self.dw=torch.FloatTensor(normal.rvs(size=[self.batch_size,self.input_dim])*np.sqrt(self.dt))
        self.hidden = [input_dim, hidden_dim, hidden_dim, input_dim]
        self.mfg = nn.ModuleList([ControlNN(self.hidden) for _ in range(self.Ntime)])
        self.linear = nn.Linear(input_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, bf):
        if bf.mode == 'forwardX':
            if bf.mf:
                return self.forwardX(bf.inputs, bf.mu)
            else:
                return self.forwardX(bf.inputs)
        elif bf.mode == 'HamGraFlow':
            return self.HamGraFlow(bf.xMat, bf.wMat, bf.loss)
        elif bf.mode == 'backwardYZ':
            if bf.mf:
                return self.backwardYZ(bf.xMat, bf.wMat, bf.loss, bf.mu)
            else:
                return self.backwardYZ(bf.xMat, bf.wMat, bf.loss)
        elif bf.mode == 'HamCompute':
            if bf.mf:
                return self.HamCompute(bf.xMat, bf.yMat, bf.zMat, bf.mu)
            else:
                return self.HamCompute(bf.xMat, bf.yMat, bf.zMat)
        elif bf.mode == 'ValueEstimator':
            return self.ValueEstimator(bf.xMat, bf.loss)
    
    # Define the forward process: Euler-Maruyama method
    def forwardX(self, x, mu=None):
        xMat = []
        wMat = []
        x0=torch.clone(x)
        xMat.append(x0)
        
        if x0.shape[0] == self.batch_size:
            dw = self.dw
        else:
            # dw=torch.FloatTensor(normal.rvs(size=x0.shape)*np.sqrt(self.dt))
            dw=torch.randn(x0.shape, device=self.device)*np.sqrt(self.dt)
            
        for i in range(self.Ntime):
            # x_mu = torch.cat((x0, mu), dim=1)
            # x0 = x0 + self.mfg_mf[i](x_mu)*self.dt + self.sigma*dw
            control = self.mfg[i](x0)
            if mu is not None:
                x0 = x0 + (self.interact*(mu - x0) + control)*self.dt + self.sigma*dw
            else:
                x0 = x0 + control*self.dt + self.sigma*dw
            xMat.append(x0)
            wMat.append(dw)
            
        # state of the batch data points
        state = x0
            
        x0=self.linear(x0)
        xMat.append(x0)
        
        x0=self.sigmoid(x0)
        xMat.append(x0)
        
        return xMat, wMat, state
    
    # Define the backward process: HJB equation to compute the optimal control loss
    def backwardYZ(self, xMat, wMat, loss_val, mu=None): 
        yMat=[]
        zMat=[]
        L=len(xMat)
        x_terminal=xMat[-1]     # The terminal X_T
        
        # This very first step is required, as it is the fully-connected layer. 
        y_terminal = autograd.grad(outputs=[loss_val], inputs=[x_terminal],
                                   grad_outputs=torch.ones_like(loss_val), allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(y_terminal)  # The terminal YT
        
        # Here y_terminal has dim batch_size x output_size (2 x 2)
        x_pro=xMat[L-2]
        hami_pro = torch.sum(y_terminal.detach()*self.sigmoid(x_pro), dim=1, keepdim=True)
        hami_pro = hami_pro.view(-1,1)
        hami_pro_x = -autograd.grad(outputs=[hami_pro], inputs=[x_pro],
                               grad_outputs=torch.ones_like(hami_pro),allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(hami_pro_x)
        
        # The last block is stand-alone FC, must be treated separately. 
        x_linear=xMat[L-3]
        hami=torch.sum(hami_pro_x.detach()*self.linear(x_linear), dim=1, keepdim=True) 
        hami=hami.view(-1,1)
        hami_x = -autograd.grad(outputs=[hami], inputs=[x_linear],
                             grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(hami_x)
        
        # finding the optimal control
        for i in range(self.Ntime-1, -1, -1):
            zMat.append(yMat[-1]*wMat[i]/self.dt)
            X=xMat[i]
            if i == 0:
                X.requires_grad = True
            control = self.mfg[i](X)
            if mu is not None:
                hami = torch.sum(yMat[-1].detach()*(self.interact*(mu.detach()-X.detach()) + control) + self.sigma*zMat[-1].detach(), dim=1, keepdim=True)
            else:
                hami = torch.sum(yMat[-1].detach()*control + self.sigma*zMat[-1].detach(), dim=1, keepdim=True)
            hami = hami.view(-1,1)
            hami_x = autograd.grad(outputs=[hami], inputs=[X],
                                   grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
            ytemp = yMat[-1] - hami_x*self.dt + zMat[-1]*wMat[i]
            yMat.append(ytemp)
                
        return yMat, zMat
    
    # Compute the Hamiltonian     H = P*F + σ*Z - L
    def HamCompute(self, xMat, yMat, zMat, mu=None):
        totalham=0.0
        L=len(xMat)
        hami_pro = torch.mean(torch.sum(yMat[0].detach()*self.sigmoid(xMat[L-2]), dim=1, keepdim=True))
        totalham += hami_pro
        hami_linear = torch.mean(torch.sum(yMat[1].detach()*self.linear(xMat[L-3]), dim=1, keepdim=True))
        totalham += hami_linear
        
        for i in range(self.Ntime):
            control = self.mfg[i](xMat[i].detach())
            if mu is not None:
                hami = torch.mean(torch.sum(yMat[len(self.mfg)+2-i-1].detach()*(self.interact*(mu.detach()-xMat[i].detach())+control) + self.sigma*zMat[i].detach(), dim=1, keepdim=True))
            else:
                hami = torch.mean(torch.sum(yMat[len(self.mfg)+2-i-1].detach()*control + self.sigma*zMat[i].detach(), dim=1, keepdim=True))
            totalham += hami
            
        return totalham/self.batch_size/self.Ntime
    
    def ValueEstimator(self, xMat, loss_val): 
        yMat=[]
        L=len(xMat)
        x_terminal=xMat[-1]     # The terminal X_T
        
        # This very first step is required, as it is the fully-connected layer. 
        y_terminal = autograd.grad(outputs=[loss_val], inputs=[x_terminal],
                                   grad_outputs=torch.ones_like(loss_val), allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(y_terminal)  # The terminal YT
        
        # Here y_terminal has dim batch_size x output_size (2 x 2)
        x_pro=xMat[L-2]
        hami_pro = torch.sum(y_terminal.detach()*self.sigmoid(x_pro), dim=1, keepdim=True)
        hami_pro = hami_pro.view(-1,1)
        hami_pro_x = autograd.grad(outputs=[hami_pro], inputs=[x_pro],
                               grad_outputs=torch.ones_like(hami_pro),allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(hami_pro_x)
        
        # The last block is stand-alone FC, must be treated separately. 
        x_linear=xMat[L-3]
        hami=torch.sum(hami_pro_x.detach()*self.linear(x_linear), dim=1, keepdim=True) 
        hami=hami.view(-1,1)
        hami_x = autograd.grad(outputs=[hami], inputs=[x_linear],
                             grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(hami_x)
            
        value = torch.mean(yMat[-1], dim=1, keepdim=True)
                
        return value
    
    def HamGraFlow(self, xMat, wMat, loss_val): 
        zMat=[]
        L=len(xMat)
        x_terminal=xMat[-1]     # The terminal X_T
        
        # This very first step is required, as it is the fully-connected layer. 
        y_terminal = autograd.grad(outputs=[loss_val], inputs=[x_terminal],
                                   grad_outputs=torch.ones_like(loss_val), allow_unused=True,retain_graph=True, create_graph=True)[0]
        
        # Here y_terminal has dim batch_size x output_size (2 x 2)
        x_pro=xMat[L-2]
        hami_pro = torch.sum(y_terminal.detach()*self.sigmoid(x_pro), dim=1, keepdim=True)
        hami_pro = hami_pro.view(-1,1)
        hami_pro_x = -autograd.grad(outputs=[hami_pro], inputs=[x_pro],
                               grad_outputs=torch.ones_like(hami_pro),allow_unused=True,retain_graph=True, create_graph=True)[0]
        
        # The last block is stand-alone FC, must be treated separately. 
        x_linear=xMat[L-3]
        hami=torch.sum(hami_pro_x.detach()*self.linear(x_linear), dim=1, keepdim=True) 
        hami=hami.view(-1,1)
        hami_x = -autograd.grad(outputs=[hami], inputs=[x_linear],
                             grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
        
        # finding the optimal control
        for i in range(self.Ntime-1, -1, -1):
            zMat.append(hami_x*wMat[i]/self.dt)
            X=xMat[i]
            if i == 0:
                X.requires_grad = True
            control = self.mfg[i](X)
            hami = torch.sum(hami_x.detach()*control + self.sigma*zMat[-1].detach(), dim=1, keepdim=True)
            hami = hami.view(-1,1)
            hami_x = -autograd.grad(outputs=[hami], inputs=[X],
                                   grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]

        return hami_x