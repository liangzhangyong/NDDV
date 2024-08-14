#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Untitled-2
@Time    :   2024/01/29 21:53:50
@Author  :   Liang Zhangyong
@Contact :   liangzhangyong1994@gmail.com
'''

# here put the import lib
import torch
 
def RSD(feature_source, feature_target, tradeoff):
    u_s, s_s, v_s = torch.svd(feature_source.t())
    u_t, s_t, v_t = torch.svd(feature_target.t())
    p_s, cospa, p_t = torch.svd(torch.mm(u_s.t(), u_t))
    sinpa = torch.sqrt(1-torch.pow(cospa,2))
    return torch.norm(sinpa,1)+tradeoff*torch.norm(torch.abs(p_s) - torch.abs(p_t), 2)