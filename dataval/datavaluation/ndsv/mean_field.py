#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2024/01/31 09:48:20
@Author  :   Liang Zhangyong
@Contact :   liangzhangyong1994@gmail.com
'''

# here put the import lib
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from sklearn.neighbors import KernelDensity

# class StateDistribution(nn.Module):
#     def __init__(self, bandwidth):
#         super(StateDistribution, self).__init__()
#         self.bandwidth = bandwidth
#         self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x_np = x.detach().cpu().numpy()
#         self.kde.fit(x_np)
#         log_density_values = self.kde.score_samples(x_np)
#         density_values = np.exp(log_density_values)
#         mu = torch.from_numpy(density_values)
#         mu = mu.view(-1, 1)
#         return mu


class StateDistribution(nn.Module):
    def __init__(self, sigma):
        super(StateDistribution, self).__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ker_x = torch.exp(-x**2 / (2 * self.sigma**2))
        mu = torch.mean(ker_x, dim=0, keepdim=True)
        mu = mu.expand(x.shape[0], -1)
        return mu