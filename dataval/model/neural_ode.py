#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2024/01/25 17:08:15
@Author  :   Liang Zhangyong
@Contact :   liangzhangyong1994@gmail.com
'''

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataval.model.api import TorchClassMixin, TorchPredictMixin

from torchdiffeq import odeint_adjoint, odeint


class ConvNeuralODE(TorchClassMixin, TorchPredictMixin):
    """ neural ode net classifier.

    Parameters
    ----------
    num_classes : int
        Number of prediction classes
    gray_scale : bool, optional
        Whether the input image is gray scaled. LeNet has been noted to not perform
        as well with color, so disable gray_scale at your own risk, by default True
    """

    def __init__(self, num_classes, img_size, num_filters=64,
                 augment_dim=0, time_dependent=False,
                 tol=1e-3, adjoint=False):
        super(ConvNeuralODE, self).__init__()
        self.img_size = img_size
        self.num_filters = num_filters
        self.augment_dim = augment_dim
        self.num_classes = num_classes
        self.flattened_dim = (img_size[0] + augment_dim) * img_size[1] * img_size[2]
        self.time_dependent = time_dependent
        self.tol = tol

        odefunc = ConvODEFunc(img_size, num_filters, augment_dim,
                              time_dependent)

        self.odeblock = ODEBlock(odefunc, is_conv=True, tol=tol,
                                 adjoint=adjoint)

        self.linear_layer = nn.Linear(self.flattened_dim, self.num_classes)

    def forward(self, x, return_features=False):
        features = self.odeblock(x)
        pred = self.linear_layer(features.view(features.size(0), -1))
        if return_features:
            return features, pred
        return pred
    

class ODEBlock(nn.Module):
    """Solves ODE defined by odefunc."""
    def __init__(self, odefunc, is_conv=False, tol=1e-3, adjoint=False):
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.is_conv = is_conv
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None):
        """Solves ODE starting from x."""
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        integration_time = torch.tensor([0, 1]).float().type_as(x)

        if self.odefunc.augment_dim > 0:
            if self.is_conv:
                # Add augmentation
                batch_size, channels, height, width = x.shape
                aug = torch.zeros(batch_size, self.odefunc.augment_dim,
                                  height, width).to(self.device)
                # Shape (batch_size, channels + augment_dim, height, width)
                x_aug = torch.cat([x, aug], 1)
            else:
                # Add augmentation
                aug = torch.zeros(x.shape[0], self.odefunc.augment_dim).to(self.device)
                # Shape (batch_size, data_dim + augment_dim)
                x_aug = torch.cat([x, aug], 1)
        else:
            x_aug = x

        if self.adjoint:
            out = odeint_adjoint(self.odefunc, x_aug, integration_time,
                                 rtol=self.tol, atol=self.tol, method='dopri5',
                                 options={'max_num_steps': 1000})
        else:
            out = odeint(self.odefunc, x_aug, integration_time,
                         rtol=self.tol, atol=self.tol, method='dopri5',
                         options={'max_num_steps': 1000})

        return out[1]  # Return only final time
    

class ConvODEFunc(nn.Module):
    """Convolutional block modeling the derivative of ODE system.
    """
    def __init__(self, img_size, num_filters, augment_dim=0,
                 time_dependent=False):
        super(ConvODEFunc, self).__init__()
        self.augment_dim = augment_dim
        self.img_size = img_size
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations
        self.channels, self.height, self.width = img_size
        self.channels += augment_dim
        self.num_filters = num_filters

        if time_dependent:
            self.conv1 = Conv2dTime(self.channels, self.num_filters,
                                    kernel_size=1, stride=1, padding=0)
            self.conv2 = Conv2dTime(self.num_filters, self.num_filters,
                                    kernel_size=3, stride=1, padding=1)
            self.conv3 = Conv2dTime(self.num_filters, self.channels,
                                    kernel_size=1, stride=1, padding=0)
        else:
            self.conv1 = nn.Conv2d(self.channels, self.num_filters,
                                   kernel_size=1, stride=1, padding=0)
            self.conv2 = nn.Conv2d(self.num_filters, self.num_filters,
                                   kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(self.num_filters, self.channels,
                                   kernel_size=1, stride=1, padding=0)

        self.act = nn.ReLU(inplace=True)

    def forward(self, t, x):
        self.nfe += 1
        if self.time_dependent:
            out = self.conv1(t, x)
            out = self.act(out)
            out = self.conv2(t, out)
            out = self.act(out)
            out = self.conv3(t, out)
        else:
            out = self.conv1(x)
            out = self.act(out)
            out = self.conv2(out)
            out = self.act(out)
            out = self.conv3(out)
        return out
    

class Conv2dTime(nn.Conv2d):
    """
    Implements time dependent 2d convolutions, by appending the time variable as
    an extra channel.
    """
    def __init__(self, in_channels, *args, **kwargs):
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        # Shape (batch_size, 1, height, width)
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        # Shape (batch_size, channels + 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)
