#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2024/01/26 20:46:54
@Author  :   Liang Zhangyong
@Contact :   liangzhangyong1994@gmail.com
'''

# here put the import lib
import torch
from torch.optim.adam import Adam

class MetaAdam(Adam):
    def __init__(self, net, *args, **kwargs):
        super(MetaAdam, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def meta_step(self, grads):
        group = self.param_groups[0]
        for i, (name, parameter) in enumerate(self.net.named_parameters()):
            if grads[i] is None:
                continue  # Skip any parameters that don't have gradients
            grad = grads[i]
            state = self.state[parameter]
            
            # Lazy state initialization
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(parameter)
                state['exp_avg_sq'] = torch.zeros_like(parameter)
            
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']
            state['step'] += 1
            
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
            
            denom = exp_avg_sq.sqrt().add_(group['eps'])
            updated_parameter = parameter - step_size * exp_avg / denom

            # Make sure to copy over the requires_grad attribute
            updated_parameter = updated_parameter.detach().requires_grad_(parameter.requires_grad)
            self.set_parameter(self.net, name, updated_parameter)