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
import numpy as np
import random
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

from dataval.datavaluation.ndsv.meta import Meta
from dataval.datavaluation.ndsv.meta_sgd import MetaSGD
from dataval.datavaluation.ndsv.meta_adam import MetaAdam
from dataval.datavaluation.ndsv.myclass import MyClass
from dataval.datavaluation.ndsv.mfg import ClassifierMFG, RegressionMFG, LogisticRegressionMFG


class NDSV(DataEvaluator, ModelMixin):

    def __init__(
        self,
        mfg_epochs: int = 100,
        base_model: str = None,
        batch_size: int = 32,
        hidden_dim: int = 25,
        Ntime: int = 2,
        totalT: float = 1.0,
        interact: int = 1,
        lr: float = 0.01,
        meta_lr: float = 1e-3,
        momentum: float = 0.9,
        dampening: float = 0.,
        weight_decay: float = 5e-4,
        nesterov: bool = False,
        meta_interval: int = 5,
        meta_weight_decay: float = 1e-4,
        meta_hidden_size: int = 10,
        meta_num_layers: int = 1,
        device: torch.device = torch.device("cpu"),
        random_state: Optional[RandomState] = None,
        lambda_RSD: float = 0.1,
        tradeoff: float = 1e-3,
        domain_adaptation: bool = False,
        bandwidth: float = 1.0,
        sigma: float = 0.01,
        re_weight: bool = True,
    ):
        # Value estimator parameters
        self.sigma = sigma
        self.re_weight = re_weight
        self.hidden_dim = hidden_dim
        self.Ntime = Ntime
        self.totalT = totalT
        self.interact=interact
        self.bandwidth = bandwidth
        self.lambda_SRD = lambda_RSD
        self.tradeoff = tradeoff
        self.domain_adaptation = domain_adaptation
        self.device = device

        # Training parameters
        self.mfg_epochs = mfg_epochs
        self.base_model = base_model
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # meta learning parameters
        self.meta_lr = meta_lr
        self.meta_interval = meta_interval
        self.meta_weight_decay = meta_weight_decay
        self.meta_hidden_size = meta_hidden_size
        self.meta_num_layers = meta_num_layers

        self.random_state = check_random_state(random_state)
        # torch.manual_seed(check_random_state(random_state).tomaxint())

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
        self.num_meta = len(x_meta)
        self.meta_dim = len(self.x_meta[0])
        self.num_classes = len(self.y_train[0])
        self.criterion = F.binary_cross_entropy if self.num_classes == 2 else F.cross_entropy
        self.myclass = MyClass()

        return self

    def train_data_values(self, *args, num_workers: int = 0, **kwargs):
        
        batch_size = min(self.batch_size, len(self.x_train))
        
        # Initial net 
        # baseline model training one epoch in one datapoint/model
        # self.mfg_net = self.pred_model.clone()
        if self.base_model.lower() == "classifiermlp":
            self.mfg_net = ClassifierMFG(self.input_dim, self.num_classes, batch_size, hidden_dim=self.hidden_dim, Ntime=self.Ntime, totalT=self.totalT, sigma=self.sigma,interact=self.interact, device=self.device).to(device=self.device)
        elif self.base_model.lower() == "regressionmlp":
            self.mfg_net = RegressionMFG(self.input_dim, self.num_classes, batch_size, hidden_dim=self.hidden_dim, Ntime=self.Ntime, totalT=self.totalT, sigma=self.sigma, interact=self.interact, device=self.device).to(device=self.device)
        elif self.base_model.lower() == "logisticregression":
            self.mfg_net = LogisticRegressionMFG(self.input_dim, self.num_classes, batch_size, hidden_dim=self.hidden_dim, Ntime=self.Ntime, totalT=self.totalT, sigma=self.sigma, interact=self.interact, device=self.device).to(device=self.device)
        meta_net = Meta(self.meta_hidden_size, self.meta_num_layers).to(device=self.device)

        # Solver
        optimizer = torch.optim.Adam(self.mfg_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # optimizer = torch.optim.SGD(self.mfg_net.parameters(), lr=self.lr, 
        #                             momentum=self.momentum, dampening=self.dampening, weight_decay=self.weight_decay, nesterov=self.nesterov)
        meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=self.meta_lr, weight_decay=self.meta_weight_decay)
        
        meta_criterion = nn.CrossEntropyLoss()
        
        # Initialize the marginal distribution
        # mu = self.mu(self.x_train)
        mu = torch.randn(self.num_points, self.input_dim, device=self.device)
        # mu = torch.mean(self.x_train, dim=0, keepdim=True)
        # mu = mu.expand(self.x_train.shape[0], -1)
        x_state = torch.zeros_like(mu)
        # x_state = torch.zeros(1, self.input_dim, device=self.device)

        # Load datapoints
        train_data = CatDataset(self.x_train, self.y_train)
        val_data = CatDataset(self.x_valid, self.y_valid)
        test_data = CatDataset(self.x_test, self.y_test)
        meta_data = CatDataset(self.x_meta, self.y_meta)
        
        train_loader = DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size, pin_memory=True)
        meta_loader = DataLoader(meta_data, batch_size, shuffle=True, pin_memory=True)
        
        meta_loader_iter = iter(meta_loader)
        
        self.epoch_train_loss = []
        self.epoch_meta_loss = []
        
        self.mfg_net.train()
        for epoch in tqdm(range(self.mfg_epochs)):
            iteration = 0
            i = 0
            total_loss = 0.0
            total_meta_loss = 0.0
            num_iter = 0
            num_meta_iter = 0
            
            # if epoch >= 0.6*self.mfg_epochs and epoch % 5 == 0:
            #     self.lr = self.lr * 0.1
            # for group in optimizer.param_groups:
            #         group['lr'] = self.lr
                    
            self.adjust_learning_rate(self.lr, optimizer, epoch, self.mfg_epochs)
            for x_batch, y_batch in train_loader:
                # Moves tensors to actual device
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)
                
                # meta learning
                if (iteration + 1) % self.meta_interval == 0:
                    pseudo_net = self.mfg_net.to(device=self.device)
                    pseudo_net.load_state_dict(self.mfg_net.state_dict())
                    pseudo_net.train()
                    pseudo_bf = self.myclass
                    pseudo_bf.mode = 'forwardX'
                    pseudo_bf.mf = False
                    pseudo_bf.inputs = x_batch
                    pseudo_out_x, pseudo_out_w, _ = pseudo_net(pseudo_bf)
                    pseudo_loss_vector = self.criterion(pseudo_out_x[-1], y_batch, reduction='none')
                    if loss_vector.dim() == 2:
                        pseudo_loss_vector_reshape = torch.reshape(pseudo_loss_vector[:,0], (-1, 1))
                    elif loss_vector.dim() == 1:
                        pseudo_loss_vector_reshape = torch.reshape(pseudo_loss_vector, (-1, 1))
                    # weight function (forward)
                    pseudo_weight = meta_net(pseudo_loss_vector_reshape.data)
                    # re-weighting loss function
                    pseudo_loss_vector_weight = pseudo_weight * pseudo_loss_vector_reshape
                    # adjoint function (backward)
                    pseudo_bf.mode = 'backwardYZ'
                    pseudo_bf.xMat = pseudo_out_x
                    pseudo_bf.wMat = pseudo_out_w
                    pseudo_bf.loss = pseudo_loss_vector_weight
                    pseudo_out_y, pseudo_out_z = pseudo_net(pseudo_bf)
                    # Hamiltonian function
                    pseudo_bf.mode = 'HamCompute'
                    pseudo_bf.xMat = pseudo_out_x
                    pseudo_bf.yMat = pseudo_out_y
                    pseudo_bf.zMat = pseudo_out_z
                    pseudo_loss = pseudo_net(pseudo_bf)

                    pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)

                    # pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=self.lr)
                    pseudo_optimizer = MetaAdam(pseudo_net, pseudo_net.parameters(), lr=self.lr)
                    pseudo_optimizer.load_state_dict(optimizer.state_dict())
                    pseudo_optimizer.meta_step(pseudo_grads)

                    del pseudo_grads, pseudo_bf, pseudo_out_x, pseudo_out_y, pseudo_out_z, pseudo_out_w
                    
                    try:
                        meta_inputs, meta_labels = next(meta_loader_iter)
                    except StopIteration:
                        meta_loader_iter = iter(meta_loader)
                        meta_inputs, meta_labels= next(meta_loader_iter)
                    meta_inputs = meta_inputs.to(device=self.device)
                    meta_labels = meta_labels.to(device=self.device)
                
                    # state meta function (forward)
                    meta_bf = self.myclass
                    meta_bf.mode = 'forwardX'
                    meta_bf.inputs = meta_inputs
                    meta_bf.mf = False
                    meta_outputs_x, meta_outputs_w, _ = pseudo_net(meta_bf)
                
                    # loss meta function
                    meta_loss_vect = meta_criterion(meta_outputs_x[-1], meta_labels)
                    
                    # adjoint meta function (backward)
                    meta_bf.mode = 'backwardYZ'
                    meta_bf.xMat = meta_outputs_x
                    meta_bf.wMat = meta_outputs_w
                    meta_bf.loss = meta_loss_vect
                    meta_outputs_y, meta_outputs_z = pseudo_net(meta_bf)
                
                    # Hamiltonian meta function
                    meta_bf.mode = 'HamCompute'
                    meta_bf.yMat = meta_outputs_y
                    meta_bf.zMat = meta_outputs_z
                    
                    meta_loss = pseudo_net(meta_bf)

                    meta_optimizer.zero_grad()
                    meta_loss.backward()
                    meta_optimizer.step()
                    
                    total_meta_loss += meta_loss.item()
                    num_meta_iter += 1
                    
                    del meta_bf, meta_outputs_w, meta_outputs_x, meta_outputs_y, meta_outputs_z
                    
                bf = self.myclass
                bf.mode = 'forwardX'
                bf.mf = True
                bf.inputs = x_batch
                bf.mu = mu[i*batch_size:min((i+1)*batch_size, len(mu))]
                outputs_x, outputs_w, outputs_state = self.mfg_net(bf)
                loss_vector = self.criterion(outputs_x[-1], y_batch, reduction='none')
                if loss_vector.dim() == 2:
                    loss_vector_reshape = torch.reshape(loss_vector[:,0], (-1, 1)) 
                elif loss_vector.dim() == 1:
                    loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))
                
                if self.re_weight:
                    with torch.no_grad():
                        weight = meta_net(loss_vector_reshape)
                    # x_state += torch.sum(weight.detach()*outputs_state.detach(), dim=0, keepdim=True)
                    x_state[i*batch_size:min((i+1)*batch_size, len(mu))] = weight.detach()*outputs_state.detach()
                    loss_vector_weight = weight * loss_vector_reshape
                else:
                    x_state[i*batch_size:min((i+1)*batch_size, len(mu))] = outputs_state.detach()
                    loss_vector_weight = loss_vector_reshape
                bf.mode = 'backwardYZ'
                bf.xMat = outputs_x
                bf.wMat = outputs_w
                bf.loss = loss_vector_weight
                outputs_y, outputs_z = self.mfg_net(bf)
                bf.mode = 'HamCompute'
                bf.xMat = outputs_x
                bf.yMat = outputs_y
                bf.zMat = outputs_z
                loss = self.mfg_net(bf)

                optimizer.zero_grad()
                loss.backward()  # Compute gradient
                optimizer.step()  # Updates weights
                
                total_loss += loss.item()
                num_iter += 1
                iteration += 1
                i += 1
                del bf, outputs_w, outputs_x, outputs_y, outputs_z
                
            # Update the marginal distribution
            mu = torch.mean(x_state, dim=0, keepdim=True)
            mu = mu.expand(x_state.shape[0], -1).to(device=self.device)
            # mu = x_state / self.num_points
            # mu = mu.expand(self.num_points, -1).to(device=self.device)
            
            avg_loss = total_loss / num_iter if num_iter > 0 else 0
            avg_meta_loss = total_meta_loss / num_meta_iter if num_meta_iter > 0 else 0
            self.epoch_train_loss.append(avg_loss)
            self.epoch_meta_loss.append(avg_meta_loss)
            
            test_accuracy = self.compute_loss_accuracy(
                net=self.mfg_net,
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
        
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device=self.device)
            y_batch = y_batch.to(device=self.device)
            
            # forward control variate
            dv = self.myclass
            dv.mode = 'forwardX'
            dv.mf = False
            dv.inputs = x_batch
            outputs_x, outputs_w, _ = self.mfg_net(dv)
            terminal_loss = self.criterion(outputs_x[-1], y_batch)
            
            dv.mode = 'HamGraFlow'
            dv.xMat = outputs_x
            dv.wMat = outputs_w
            dv.loss = terminal_loss
            # batch_data_values = self.mfg_net(dv)
            
            # control cost at data gradient to estimate data value
            batch_data_values = self.data_value_estimator(terminal_loss, outputs_x[-1])
            data_values = torch.cat([data_values, batch_data_values])
            
        # data_values = self.normalize_gradients(data_values)
        data_values = data_values.squeeze()
        num_points = len(data_values) - 1
        data_values = data_values * (1 + 1 / (num_points)) - data_values.sum() / num_points
        # data_values = self.normalize_gradients(data_values)
        
        return data_values.numpy(force=True)
    
    
    def data_values_trajectory(self):
        data = CatDataset(self.x_train, self.y_train)
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)
        
        trajectories = []
        time_span = np.linspace(0.0, 1.0, self.Ntime+1)
        time_span = torch.from_numpy(time_span).to(self.device)
    
        for x_batch, y_batch in data_loader:
            batch_data_values = []
            x_batch = x_batch.to(device=self.device)
            y_batch = y_batch.to(device=self.device)
            
            dvt = self.myclass
            dvt.mode = 'forwardX'
            dvt.mf = False
            dvt.inputs = x_batch
            outputs_x, _, _ = self.mfg_net(dvt)
            for i in range(self.Ntime+1):
                dvt.mode = 'stateExtract'
                dvt.x = outputs_x[i]
                output = self.mfg_net(dvt)
                terminal_loss = self.criterion(output, y_batch)
                batch_data_values.append(self.data_value_estimator(terminal_loss, output))
                
            traj = torch.stack(batch_data_values, dim=0).cpu().detach().numpy()
            trajectories.append(traj)
            
        trajectories = np.concatenate(trajectories, 1)
        num_points = trajectories.shape[1] - 1
        sum_values = trajectories.sum(axis=1, keepdims=True)
        trajectories = trajectories * (1 + 1 / num_points) - sum_values / num_points
        
            
        return time_span, trajectories
    
    
    def data_state_trajectory(self):
        data = CatDataset(self.x_train, self.y_train)
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)
        
        trajectories = []
        outputs = []
        classes = []
        time_span = np.linspace(0.0, 1.0, self.Ntime+1)
        time_span = torch.from_numpy(time_span).to(self.device)
        
        self.mfg_net.eval()
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)
                
                dst = self.myclass
                dst.mode = 'forwardX'
                dst.mf = False
                dst.inputs = x_batch
                outputs_x, _, _ = self.mfg_net(dst)
                outputs = [torch.mean(out, dim=1, keepdim=True) for out in outputs_x]
                traj = torch.stack(outputs[:self.Ntime+1], dim=0).cpu().numpy()
                trajectories.append(traj)
                classes.extend(y_batch.cpu().numpy())
                
            trajectories = np.concatenate(trajectories, 1)
        
        return time_span, trajectories
    
    
    def data_value_estimator(self, cost, x_terminal):
        data_values = -autograd.grad(outputs=[cost], inputs=[x_terminal],
                                    grad_outputs=torch.ones_like(cost),allow_unused=True,retain_graph=True, create_graph=True)[0]
        
        data_values = x_terminal * data_values
        data_values = torch.mean(data_values, dim=1, keepdim=True)
        return data_values
    
    
    def adjust_learning_rate(self, lr, optimizer, epochs, max_epoch):
        lr = lr * ((0.1 ** int(epochs >= 0.6*max_epoch)) * (0.1 ** int(epochs >= 0.8*max_epoch)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
            
    def normalize_gradients(self, gradients, new_min=-1, new_max=1):
        min_val = torch.min(gradients)
        max_val = torch.max(gradients)
        if min_val == max_val:
            return torch.full_like(gradients, new_min)
        normalized_gradients = (gradients - min_val) / (max_val - min_val)
        normalized_gradients = normalized_gradients * (new_max - new_min) + new_min
        
        return normalized_gradients
    
    
    def save_train_meta_loss(self):
        
        return self.epoch_train_loss, self.epoch_meta_loss
        # import csv
        # # write the loss to csv
        # with open('./results_loss/training_losses_2dplanes.csv', 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(['Epoch', 'Loss', 'Meta_Loss'])
        #     for epoch, (loss, meta_loss) in enumerate(zip(self.epoch_loss, self.epoch_meta_loss)):
        #         writer.writerow([epoch, loss, meta_loss])