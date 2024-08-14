#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2024/01/26 20:40:59
@Author  :   Liang Zhangyong
@Contact :   liangzhangyong1994@gmail.com
'''

# here put the import lib
import torch
import torch.nn as nn


# Gauss RBF Meta
class GaussianKernel(nn.Module):
    def __init__(self, in_features, out_features, grid_min, grid_max, num_grids, spline_scale):
        super(GaussianKernel, self).__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.grid = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids), requires_grad=False)
        self.spline_weight = nn.Parameter(torch.randn(in_features * num_grids, out_features) * spline_scale)

    def forward(self, x):
        x = x.unsqueeze(-1)
        basis = torch.exp(-((x - self.grid) / ((self.grid_max - self.grid_min) / (self.num_grids - 1))) ** 2)
        return basis.view(basis.size(0), -1).matmul(self.spline_weight)
    
    
class Matern12Kernel(nn.Module):
    def __init__(self, in_features, out_features, grid_min, grid_max, num_grids, spline_scale):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.grid = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids), requires_grad=True)
        self.spline_weight = nn.Parameter(torch.randn(in_features * num_grids, out_features) * spline_scale)

    def forward(self, x):
        x = x.unsqueeze(-1)
        dist = torch.abs(x - self.grid)
        scale = (self.grid_max - self.grid_min) / (self.num_grids - 1)
        basis = torch.exp(-dist / scale)
        return basis.view(basis.size(0), -1).matmul(self.spline_weight)
    

class Matern32Kernel(nn.Module):
    def __init__(self, in_features, out_features, grid_min, grid_max, num_grids, spline_scale):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.grid = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids), requires_grad=True)
        self.spline_weight = nn.Parameter(torch.randn(in_features*num_grids, out_features) * spline_scale)

    def forward(self, x):
        x = x.unsqueeze(-1)
        sqrt3 = torch.sqrt(torch.tensor(3.0))
        dist = torch.abs(x - self.grid)
        scale_dist = sqrt3 * dist / ((self.grid_max - self.grid_min) / (self.num_grids - 1))
        basis = (1 + scale_dist) * torch.exp(-scale_dist)
        return basis.view(basis.size(0), -1).matmul(self.spline_weight)


class Matern52Kernel(nn.Module):
    def __init__(self, in_features, out_features, grid_min, grid_max, num_grids, spline_scale):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.grid = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids), requires_grad=True)
        self.spline_weight = nn.Parameter(torch.randn(in_features*num_grids, out_features) * spline_scale)

    def forward(self, x):
        x = x.unsqueeze(-1)
        sqrt5 = torch.sqrt(torch.tensor(5.0))
        dist = torch.abs(x - self.grid)
        scale_dist = sqrt5 * dist / ((self.grid_max - self.grid_min) / (self.num_grids - 1))
        basis = (1 + scale_dist + (5 * dist**2) / (3 * ((self.grid_max - self.grid_min) / (self.num_grids - 1))**2)) * torch.exp(-scale_dist)
        return basis.view(basis.size(0), -1).matmul(self.spline_weight)
    

class MetaInter(nn.Module):
    def __init__(
        self, 
        hidden_size=100, 
        grid_min=-2., 
        grid_max=2., 
        num_grids=8, 
        spline_scale=0.1,
        kernel='Gaussian'):
        super(MetaInter, self).__init__()
        self.hidden_size = hidden_size
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.spline_scale = spline_scale
        self.kernel = kernel
        self.base_layer = nn.Linear(1, self.hidden_size)
        self.linear_layer = nn.Linear(self.hidden_size, 1)
        self.act = nn.SiLU()
        if self.kernel == 'Gaussian':
            self.kan_layer = GaussianKernel(1, self.hidden_size, self.grid_min, self.grid_max, self.num_grids, self.spline_scale)
        elif self.kernel == 'Matern12':
            self.kan_layer = Matern12Kernel(1, self.hidden_size, self.grid_min, self.grid_max, self.num_grids, self.spline_scale)
        elif self.kernel == 'Matern32':
            self.kan_layer = Matern32Kernel(1, self.hidden_size, self.grid_min, self.grid_max, self.num_grids, self.spline_scale)
        elif self.kernel == 'Matern52':
            self.kan_layer = Matern52Kernel(1, self.hidden_size, self.grid_min, self.grid_max, self.num_grids, self.spline_scale)

    def forward(self, x):
        ret = self.kan_layer(x)
        base = self.base_layer(self.act(x))
        x = ret + base
        x = self.linear_layer(x)
        return torch.sigmoid(x)
    
    
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
    
    
# class MetaInter(nn.Module):
#     def __init__(self, hidden_size=100, grid_min=-1., grid_max=1., num_grids=8, spline_scale=0.1):
#         super(MetaInter, self).__init__()
#         self.rbf_layer = GRBFKANLayer(1, hidden_size, grid_min, grid_max, num_grids, spline_scale)
#         self.base_layer = nn.Linear(1, hidden_size)
#         self.linear_layer = nn.Linear(hidden_size, 1)
#         self.act = nn.SiLU()

#     def forward(self, x):
#         ret = self.rbf_layer(x)
#         base = self.base_layer(self.act(x))
#         x = ret + base
#         x = self.linear_layer(x)
#         return torch.sigmoid(x)