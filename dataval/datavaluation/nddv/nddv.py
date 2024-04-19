#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ame.py
@Time    :   2024/01/15 22:20:44
@Author  :   Liang Zhangyong
@Contact :   liangzhangyong1994@gmail.com
'''

# here put the import lib
from collections import OrderedDict
from typing import Optional

# import signatory as sg
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from tqdm import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader

from dataval.dataloader.util import CatDataset
from dataval.datavaluation.api import DataEvaluator, ModelMixin
from dataval.util import set_random_state

from dataval.datavaluation.nddv.myclass import MyClass
from dataval.datavaluation.nddv.smp_oc import ClassifierSMP, RegressionSMP, LogisticRegressionSMP


class NDDV(DataEvaluator, ModelMixin):

    def __init__(
        self,
        max_epochs: int = 100,
        base_model: str = None,
        batch_size: int = 32,
        lr: float = 0.01,
        Ntime: int = 2,
        totalT: float = 1.0,
        momentum: float = 0.9,
        dampening: float = 0.,
        weight_decay: float = 5e-4,
        nesterov: bool = False,
        device: torch.device = torch.device("cpu"),
        random_state: Optional[RandomState] = None,
    ):
        # Value estimator parameters
        self.device = device

        # Training parameters
        self.max_epochs = max_epochs
        self.base_model = base_model
        self.batch_size = batch_size
        self.lr = lr
        self.Ntime = Ntime
        self.totalT = totalT
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # self.random_state = check_random_state(random_state)
        torch.manual_seed(check_random_state(random_state).tomaxint())

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x_meta: torch.Tensor,
        y_meta: torch.Tensor,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test
        self.x_meta = x_meta
        self.y_meta = y_meta
        
        self.num_points = len(x_train)
        self.input_dim = len(self.x_train[0])
        self.num_classes = len(self.y_train[0])
        self.criterion = F.binary_cross_entropy if self.num_classes == 2 else F.cross_entropy
        self.myclass = MyClass()

        return self

    def train_data_values(self, *args, num_workers: int = 0, **kwargs):
        batch_size = min(self.batch_size, len(self.x_train))
        
        # Initial net 
        # baseline model training one epoch in one datapoint/model
        if self.base_model.lower() == "classifiermlp":
            self.smp_oc = ClassifierSMP(self.input_dim, self.num_classes, batch_size, Ntime=self.Ntime, totalT=self.totalT, device=self.device).to(device=self.device)
        elif self.base_model.lower() == "regressionmlp":
            self.smp_oc = RegressionSMP(self.input_dim, self.num_classes, batch_size, Ntime=self.Ntime, totalT=self.totalT, device=self.device).to(device=self.device)
        elif self.base_model.lower() == "logisticregression":
            self.smp_oc = LogisticRegressionSMP(self.input_dim, self.num_classes, batch_size, Ntime=self.Ntime, totalT=self.totalT, device=self.device).to(device=self.device)

        # Solver
        optimizer = torch.optim.Adam(self.smp_oc.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.smp_oc.parameters(), lr=self.lr, 
        #                             momentum=self.momentum, dampening=self.dampening, weight_decay=self.weight_decay, nesterov=self.nesterov)
        
        # Initialize the marginal distribution
        mu = torch.randn(self.num_points, self.input_dim, device=self.device)
        # mu = torch.mean(self.x_train, dim=0, keepdim=True)
        # mu = mu.expand(self.x_train.shape[0], -1)
        x_state = torch.zeros_like(mu)
        # x_state = torch.zeros(1, self.input_dim, device=self.device)

        # Load datapoints
        train_data = CatDataset(self.x_train, self.y_train)
        val_data = CatDataset(self.x_valid, self.y_valid)
        test_data = CatDataset(self.x_test, self.y_test)
        
        train_loader = DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size, pin_memory=True)
        
        self.smp_oc.train()
        for epoch in tqdm(range(self.max_epochs)):
            iteration = 0
            i = 0
            
            # self.adjust_learning_rate(self.lr, optimizer, epoch, self.max_epochs)
            if epoch >= 0.6*self.max_epochs and epoch % 5 == 0:
                self.lr = self.lr * 0.1
            for group in optimizer.param_groups:
                    group['lr'] = self.lr
                
            for x_batch, y_batch in train_loader:
                # Moves tensors to actual device
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)
                        
                bf = self.myclass
                bf.mode = 'forwardX'
                bf.mf = True
                bf.inputs = x_batch
                bf.mu = mu[i*batch_size:min((i+1)*batch_size, len(mu))]
                outputs_x, outputs_w, outputs_state = self.smp_oc(bf)
                loss_vector = self.criterion(outputs_x[-1], y_batch, reduction='none')
                if loss_vector.dim() == 2:
                    loss_vector_reshape = torch.reshape(loss_vector[:,0], (-1, 1)) 
                elif loss_vector.dim() == 1:
                    loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))
            
                bf.mode = 'HamGraFlow'
                bf.xMat = outputs_x
                bf.wMat = outputs_w
                bf.loss = loss_vector_reshape
                hami_flow = self.smp_oc(bf)
                # Hamilton gradient flow
                weight = torch.mean(hami_flow, dim=1, keepdim=True)
                weight = self.normalize_gradients(weight)
                
                # reweighting loss function  
                loss_vector_weight = weight * loss_vector_reshape
                x_state[i*batch_size:min((i+1)*batch_size, len(mu))] = weight.detach()*outputs_state.detach()
                # x_state += torch.sum(weight.detach()*outputs_state.detach(), dim=0, keepdim=True)
                
                bf.mode = 'backwardYZ'
                bf.weight_loss = loss_vector_weight
                outputs_y, outputs_z = self.smp_oc(bf)
                bf.mode = 'HamCompute'
                bf.xMat = outputs_x
                bf.yMat = outputs_y
                bf.zMat = outputs_z
                loss = self.smp_oc(bf)

                optimizer.zero_grad()
                loss.backward()  # Compute gradient
                optimizer.step()  # Updates weights
                iteration += 1
                i += 1
                
            # Update the marginal distribution
            mu = torch.mean(x_state, dim=0, keepdim=True)
            mu = mu.expand(x_state.shape[0], -1).to(device=self.device)
            # mu = x_state / self.num_points
            # mu = mu.expand(self.num_points, -1).to(device=self.device)
            
            test_accuracy = self.compute_loss_accuracy(
                net=self.smp_oc,
                inputs=self.x_test,
                labels=self.y_test,
            )
            
            print('Epoch: {}, Test Accuracy: {:.2%}'.format(
                epoch,
                test_accuracy,
            ))
            
        self.mu = mu

        return self
    
    
    def compute_loss_accuracy(self, net, inputs, labels):
        from dataval.metrics import Metrics
        from torch.utils.data import DataLoader, Dataset
        metric = Metrics("accuracy")
        if isinstance(inputs, Dataset):
            inputs = next(iter(DataLoader(inputs, batch_size=len(inputs), pin_memory=True)))
        inputs = inputs.to(device=self.device)
        
        net.eval()
        with torch.no_grad():
            bf = self.myclass
            bf.mode = 'forwardX'
            bf.inputs = inputs
            bf.mf = False
            y_hat, _, _ = net(bf)
            
        perf = metric(labels, y_hat[-1].cpu())

        return perf
    

    def evaluate_data_values(self) -> np.ndarray:
        data = CatDataset(self.x_train, self.y_train)
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)

        # Estimates data value
        data_values = torch.zeros(0, 1, device=self.device)
        i = 0
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device=self.device)
            y_batch = y_batch.to(device=self.device)
            
            # forward control variate
            dv = self.myclass
            dv.mode = 'forwardX'
            dv.mf = False
            dv.inputs = x_batch
            outputs_x, outputs_w, _ = self.smp_oc(dv)
            terminal_loss = self.criterion(outputs_x[-1], y_batch)
            
            # FBSDE to estimate data value
            dv.mode = 'ValueEstimator'
            dv.xMat = outputs_x
            dv.wMat = outputs_w
            dv.loss = terminal_loss
            # batch_data_values = self.smp_oc(dv)
            
            # control cost at data gradient to estimate data value
            batch_data_values = self.data_value_estimator(terminal_loss, outputs_x[-1])
            data_values = torch.cat([data_values, batch_data_values])
            i += 1
            
        # data_values = self.normalize_gradients(data_values)
        data_values = data_values.squeeze()
        num_points = len(data_values) - 1
        data_values = data_values * (1 + 1 / (num_points)) - data_values.sum() / num_points
        
        return data_values.numpy(force=True)
    
    
    def data_state_trajectory(self):
        data = CatDataset(self.x_train, self.y_train)
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)
        
        trajectories = []
        outputs = []
        classes = []
        time_span = np.linspace(0.0, 1.0, self.Ntime+1)
        time_span = torch.from_numpy(time_span).to(self.device)
        
        self.smp_oc.eval()
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)
                
                dst = self.myclass
                dst.mode = 'forwardX'
                dst.mf = False
                dst.inputs = x_batch
                outputs_x, _, _ = self.smp_oc(dst)
                outputs = [torch.mean(out, dim=1, keepdim=True) for out in outputs_x]
                traj = torch.stack(outputs[:self.Ntime+1], dim=0).cpu().numpy()
                trajectories.append(traj)
                classes.extend(y_batch.cpu().numpy())
                
            trajectories = np.concatenate(trajectories, 1)
        
        return time_span, trajectories
    
    
    def data_values_trajectory(self):
        pass
    
    def save_train_meta_loss(self):
        pass
    
    
    def data_value_estimator(self, cost, x_terminal):
        data_values = -autograd.grad(outputs=[cost], inputs=[x_terminal],
                                    grad_outputs=torch.ones_like(cost),allow_unused=True,retain_graph=True, create_graph=True)[0]
        data_values = x_terminal * data_values
        data_values = torch.mean(data_values, dim=1, keepdim=True)
        return data_values
    
    
    def adjust_learning_rate(self, lr, optimizer, epochs, max_epoch):
        lr = lr * ((0.1 ** int(epochs >= 0.4*max_epoch)) * (0.1 ** int(epochs >= 0.8*max_epoch)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
            
    def normalize_gradients(self, gradients, new_min=0, new_max=1):
        min_val = torch.min(gradients)
        max_val = torch.max(gradients)
        if min_val == max_val:
            return torch.full_like(gradients, new_min)
        normalized_gradients = (gradients - min_val) / (max_val - min_val)
        normalized_gradients = normalized_gradients * (new_max - new_min) + new_min
        
        return normalized_gradients