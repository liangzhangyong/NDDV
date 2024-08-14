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
        
        
class HalfSwish(torch.nn.Module):
    def __init__(self):
        super(HalfSwish, self).__init__()
        self.silu = torch.nn.SiLU()

    def forward(self, x):
        return torch.where(x >= 0, self.silu(x), torch.zeros_like(x))
 
    
# MLP for control
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
        # self.act = nn.SiLU()
        # self.act = nn.Hardswish()
        self.act = nn.ELU()
        # self.act = HalfSwish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activate:
            return self.act(self.linear(x))
        else:
            return self.linear(x)
 
        
class ControlMLP(nn.Module):
    def __init__(self, num_hiddens: list[int]):
        super(ControlMLP, self).__init__()
        self.bn = nn.BatchNorm1d(num_hiddens[0])
        self.layers = [Dense(num_hiddens[i-1],num_hiddens[i]) for i in range(1, len(num_hiddens)-1)]
        self.layers += [Dense(num_hiddens[-2], num_hiddens[-1],activate=False)]
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.bn(x))
        

# KAN for control
# class RadialBasisFunction(nn.Module):
#     def __init__(
#         self,
#         grid_min: float = -2.,
#         grid_max: float = 2.,
#         num_grids: int = 8,
#     ):
#         super().__init__()
#         grid = torch.linspace(grid_min, grid_max, num_grids)
#         self.grid = torch.nn.Parameter(grid, requires_grad=True)

#     def forward(self, x):
#         return torch.exp(-(x[..., None] - self.grid) ** 2)

    
# class GRBFKANLayer(nn.Module):
#     def __init__(
#         self, 
#         input_dim: int,
#         output_dim: int,
#         grid_min: float,
#         grid_max: float,
#         num_grids: int,
#         spline_scale: float,
#         activate: bool = True,
#     ):
#         super().__init__()
#         self.layernorm = nn.LayerNorm(input_dim)
#         self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
#         self.activate = activate
#         self.act = nn.SiLU()
#         self.base_linear = nn.Linear(input_dim, output_dim)
#         self.spline_weight = torch.nn.Parameter(torch.Tensor(output_dim, input_dim, num_grids))
#         nn.init.trunc_normal_(self.spline_weight, mean=0.0, std=spline_scale)

#     def forward(self, x):
#         base = self.base_linear(self.act(x))
#         spline_basis = self.rbf(self.layernorm(x))
#         spline = torch.einsum(
#             "...in,oin->...o", spline_basis, self.spline_weight)
#         return base + spline


class GaussianKernel(nn.Module):
    def __init__(self,in_state,out_state,grid_min,grid_max,num_grids,spline_scale):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.grid = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids), requires_grad=True)
        self.spline_weight = nn.Parameter(torch.randn(in_state*num_grids, out_state)* spline_scale)

    def forward(self, x):
        x = x.unsqueeze(-1)
        basis = torch.exp(-((x - self.grid) / ((self.grid_max - self.grid_min) / (self.num_grids - 1))) ** 2)
        return basis.view(basis.size(0), -1).matmul(self.spline_weight)


#### MaternKAN
class Matern12Kernel(nn.Module):
    def __init__(self, in_state, out_state, grid_min, grid_max, num_grids, spline_scale):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.grid = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids), requires_grad=True)
        self.spline_weight = nn.Parameter(torch.randn(in_state * num_grids, out_state) * spline_scale)

    def forward(self, x):
        x = x.unsqueeze(-1)
        dist = torch.abs(x - self.grid)
        scale = (self.grid_max - self.grid_min) / (self.num_grids - 1)
        basis = torch.exp(-dist / scale)
        return basis.view(basis.size(0), -1).matmul(self.spline_weight)
    

class Matern32Kernel(nn.Module):
    def __init__(self, in_state, out_state, grid_min, grid_max, num_grids, spline_scale):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.grid = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids), requires_grad=True)
        self.spline_weight = nn.Parameter(torch.randn(in_state*num_grids, out_state) * spline_scale)

    def forward(self, x):
        x = x.unsqueeze(-1)
        sqrt3 = torch.sqrt(torch.tensor(3.0))
        dist = torch.abs(x - self.grid)
        scale_dist = sqrt3 * dist / ((self.grid_max - self.grid_min) / (self.num_grids - 1))
        basis = (1 + scale_dist) * torch.exp(-scale_dist)
        return basis.view(basis.size(0), -1).matmul(self.spline_weight)

class Matern52Kernel(nn.Module):
    def __init__(self, in_state, out_state, grid_min, grid_max, num_grids, spline_scale):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.grid = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids), requires_grad=True)
        self.spline_weight = nn.Parameter(torch.randn(in_state*num_grids, out_state) * spline_scale)

    def forward(self, x):
        x = x.unsqueeze(-1)
        sqrt5 = torch.sqrt(torch.tensor(5.0))
        dist = torch.abs(x - self.grid)
        scale_dist = sqrt5 * dist / ((self.grid_max - self.grid_min) / (self.num_grids - 1))
        basis = (1 + scale_dist + (5 * dist**2) / (3 * ((self.grid_max - self.grid_min) / (self.num_grids - 1))**2)) * torch.exp(-scale_dist)
        return basis.view(basis.size(0), -1).matmul(self.spline_weight)


class KANLayer(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        grid_min: float,
        grid_max: float,
        num_grids: int,
        spline_scale: float,
        activate: bool = True,
        kernel: str = 'Gaussian',
    ):
        super(KANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activate = activate
        self.act = nn.SiLU()
        self.kernel = kernel
        if self.kernel == 'Matern52':
            self.kan_linear = Matern52Kernel(input_dim, output_dim, grid_min, grid_max, num_grids, spline_scale)
        elif self.kernel == 'Matern32':
            self.kan_linear = Matern32Kernel(input_dim, output_dim, grid_min, grid_max, num_grids, spline_scale)
        elif self.kernel == 'Matern12':
            self.kan_linear = Matern12Kernel(input_dim, output_dim, grid_min, grid_max, num_grids, spline_scale)
        elif self.kernel == 'Gaussian':
            self.kan_linear = GaussianKernel(input_dim, output_dim, grid_min, grid_max, num_grids, spline_scale)
        self.base_linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        ret = self.kan_linear(x)
        if self.activate:
            base = self.base_linear(self.act(x))
        else:
            base = self.base_linear(x)
        return ret + base
    
    
class ControlKAN(nn.Module):
    def __init__(self, num_hiddens: list[int], grid_min: float, grid_max: float, num_grids: int, spline_scale: float, kernel: str):
        super(ControlKAN, self).__init__()
        self.layers = [KANLayer(num_hiddens[i-1], num_hiddens[i], grid_min, grid_max, num_grids, spline_scale, kernel) for i in range(1, len(num_hiddens))]
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)
    

# Valuation model by classificaiton 
class ClassifierMFG(nn.Module):
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
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        spline_scale: float = 0.1,
        act_fn: Optional[Callable] = None,
        interpret: bool = False,
        kernel: str = 'Gaussian',
        device: torch.device = torch.device("cpu"),
    ):
        super(ClassifierMFG, self).__init__()
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
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.spline_scale = spline_scale
        self.kernel = kernel
        # self.dw=torch.FloatTensor(normal.rvs(size=[self.batch_size,self.input_dim])*np.sqrt(self.dt))
        self.dw = torch.randn(self.batch_size,self.input_dim, device=self.device)*np.sqrt(self.dt)
        self.hidden = [input_dim, hidden_dim, hidden_dim, input_dim]
        if interpret:
            self.mfg = nn.ModuleList([ControlKAN(self.hidden,self.grid_min,self.grid_max,self.num_grids,self.spline_scale,self.kernel) for _ in range(self.Ntime)])
        else:
            self.mfg = nn.ModuleList([ControlMLP(self.hidden) for _ in range(self.Ntime)])
        self.linear = nn.Linear(input_dim, num_classes)
        self.softmax = nn.Softmax(-1)
        # self.apply(_weights_init)
        
    def forward(self, bf):
        if bf.mode == 'forwardX':
            if bf.mf:
                return self.forwardX(bf.inputs, bf.mu)
            else:
                return self.forwardX(bf.inputs)
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
        elif bf.mode == 'HamGraFlow':
            return self.HamGraFlow(bf.xMat, bf.wMat, bf.loss)
        elif bf.mode == 'stateExtract':
            return self.stateExtract(bf.x)
    
    def stateExtract(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        
        return x
    
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
            
        return totalham/self.batch_size/(self.Ntime+2)
    
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
            
        data_values = x_linear * data_values
        data_values = torch.mean(data_values, dim=1, keepdim=True)
                
        return data_values
    
    def HamGraFlow(self, xMat, wMat, loss_val):
        yMat=[]
        zMat=[]
        data_values=[]
        L=len(xMat)
        x_terminal=xMat[-1]     # The terminal X_T
        
        # This very first step is required, as it is the fully-connected layer. 
        y_terminal = autograd.grad(outputs=[loss_val], inputs=[x_terminal],
                                   grad_outputs=torch.ones_like(loss_val), allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(y_terminal)  # The terminal YT
        
        values_term = -x_terminal*y_terminal
        values_term = torch.mean(values_term, dim=1, keepdim=True)
        data_values.append(values_term)
        
        # Here y_terminal has dim batch_size x output_size (2 x 2)
        x_pro=xMat[L-2]
        hami_pro = torch.sum(y_terminal.detach()*self.softmax(x_pro), dim=1, keepdim=True)
        hami_pro = hami_pro.view(-1,1)
        hami_pro_x = -autograd.grad(outputs=[hami_pro], inputs=[x_pro],
                               grad_outputs=torch.ones_like(hami_pro),allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(hami_pro_x)
        
        values_pro = -x_pro*yMat[-1]
        values_pro = torch.mean(values_pro, dim=1, keepdim=True)
        data_values.append(values_pro)
        
        # The last block is stand-alone FC, must be treated separately. 
        x_linear=xMat[L-3]
        hami=torch.sum(hami_pro_x.detach()*self.linear(x_linear), dim=1, keepdim=True) 
        hami=hami.view(-1,1)
        hami_x = -autograd.grad(outputs=[hami], inputs=[x_linear],
                             grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
        yMat.append(hami_x)
        
        values_linear = -x_linear*yMat[-1]
        values_linear = torch.mean(values_linear, dim=1, keepdim=True)
        data_values.append(values_linear)
        
        # finding the optimal control
        for i in range(self.Ntime-1, -1, -1):
            zMat.append(yMat[-1]*wMat[i]/self.dt)
            X=xMat[i]
            if i == 0:
                X.requires_grad = True
            control = self.mfg[i](X)
            hami = torch.sum(yMat[-1].detach()*control + self.sigma*zMat[-1].detach(), dim=1, keepdim=True)
            hami = hami.view(-1,1)
            hami_x = autograd.grad(outputs=[hami], inputs=[X],
                                   grad_outputs=torch.ones_like(hami),allow_unused=True,retain_graph=True, create_graph=True)[0]
            ytemp = yMat[-1] - hami_x*self.dt + zMat[-1]*wMat[i]
            yMat.append(ytemp)
            values = -X*yMat[-1]
            values = torch.mean(values, dim=1, keepdim=True)
            data_values.append(values)
        
        return data_values 
    

# Valuation model by regression    
class RegressionMFG(nn.Module):
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
        super(RegressionMFG, self).__init__()
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
        self.mfg = nn.ModuleList([ControlMLP(self.hidden) for _ in range(self.Ntime)])
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, bf):
        if bf.mode == 'forwardX':
            if bf.mf:
                return self.forwardX(bf.inputs, bf.mu)
            else:
                return self.forwardX(bf.inputs)
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
        elif bf.mode == 'HamGraFlow':
            return self.HamGraFlow(bf.xMat, bf.wMat, bf.loss)
    
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
        data_values = X * hami_x
        data_values = torch.mean(data_values, dim=1, keepdim=True)
        
        return data_values 
    

# Valuation model by logistic regression    
class LogisticRegressionMFG(nn.Module):
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
        super(LogisticRegressionMFG, self).__init__()
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
        self.mfg = nn.ModuleList([ControlMLP(self.hidden) for _ in range(self.Ntime)])
        self.linear = nn.Linear(input_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, bf):
        if bf.mode == 'forwardX':
            if bf.mf:
                return self.forwardX(bf.inputs, bf.mu)
            else:
                return self.forwardX(bf.inputs)
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
        elif bf.mode == 'HamGraFlow':
            return self.HamGraFlow(bf.xMat, bf.wMat, bf.loss)
        elif bf.mode == 'stateExtract':
            return self.stateExtract(bf.x)
    
    def stateExtract(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        
        return x
    
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
            
        return totalham/self.batch_size/(self.Ntime+2)
    
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

        data_values = X * hami_x
        data_values = torch.mean(data_values, dim=1, keepdim=True)
        
        return data_values 