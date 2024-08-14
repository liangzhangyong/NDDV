#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2024/01/26 20:46:54
@Author  :   Liang Zhangyong
@Contact :   liangzhangyong1994@gmail.com
'''

# here put the import lib
from torch.optim.sgd import SGD


class MetaSGD(SGD):
    def __init__(self, net, *args, **kwargs):
        super(MetaSGD, self).__init__(*args, **kwargs)
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
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        lr = group['lr']

        # Instead of detaching the parameter, we clone it
        # The clone will have requires_grad=True if the original parameter did
        for (name, parameter), grad in zip(self.net.named_parameters(), grads):
            if grad is None:
                continue  # Skip any parameters that don't have gradients
            
            # Apply weight decay directly to the grad if necessary
            if weight_decay != 0:
                grad = grad.add(parameter, alpha=weight_decay)
            
            # Apply momentum directly to the grad if necessary
            if momentum != 0:
                if 'momentum_buffer' not in self.state[parameter]:
                    buffer = grad
                else:
                    buffer = self.state[parameter]['momentum_buffer']
                buffer = buffer.mul(momentum).add(grad, alpha=1-dampening)
                self.state[parameter]['momentum_buffer'] = buffer
                if nesterov:
                    grad = grad.add(buffer, alpha=momentum)
                else:
                    grad = buffer
            
            # Instead of updating the parameter in-place, clone, update and replace
            updated_parameter = parameter - lr * grad
            
            # Make sure to copy over the requires_grad attribute
            updated_parameter = updated_parameter.clone().detach().requires_grad_(parameter.requires_grad)
            self.set_parameter(self.net, name, updated_parameter)
            


# class MetaSGD(SGD):
#     def __init__(self, net, *args, **kwargs):
#         super(MetaSGD, self).__init__(*args, **kwargs)
#         self.net = net

#     def set_parameter(self, current_module, name, parameters):
#         if '.' in name:
#             name_split = name.split('.')
#             module_name = name_split[0]
#             rest_name = '.'.join(name_split[1:])
#             for children_name, children in current_module.named_children():
#                 if module_name == children_name:
#                     self.set_parameter(children, rest_name, parameters)
#                     break
#         else:
#             current_module._parameters[name] = parameters

#     def meta_step(self, grads):
#         group = self.param_groups[0]
#         weight_decay = group['weight_decay']
#         momentum = group['momentum']
#         dampening = group['dampening']
#         nesterov = group['nesterov']
#         lr = group['lr']

#         for (name, parameter), grad in zip(self.net.named_parameters(), grads):
#             parameter.detach_()
#             if weight_decay != 0:
#                 grad_wd = grad.add(parameter, alpha=weight_decay)
#             else:
#                 grad_wd = grad
#             if momentum != 0 and 'momentum_buffer' in self.state[parameter]:
#                 buffer = self.state[parameter]['momentum_buffer']
#                 grad_b = buffer.mul(momentum).add(grad_wd, alpha=1-dampening)
#             else:
#                 grad_b = grad_wd
#             if nesterov:
#                 grad_n = grad_wd.add(grad_b, alpha=momentum)
#             else:
#                 grad_n = grad_b
#             self.set_parameter(self.net, name, parameter.add(grad_n, alpha=-lr))