#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import numpy as np



class MOSLoss(nn.Module):
    def __init__(self, n_classes, ignore_index):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.softmax = nn.Softmax(dim=1)
        weight = [0.0 if i in ignore_index else 1.0 for i in range(n_classes)]

        weight = torch.Tensor([w / sum(weight) for w in weight])
        self.loss = nn.NLLLoss(weight=weight)

    def compute_loss(self, out, past_labels):
        # Get raw point wise scores
        logits = out

        # Set ignored classes to -inf to not influence softmax
        logits[:, self.ignore_index] = -float("inf")

        softmax = self.softmax(logits)
        log_softmax = torch.log(softmax.clamp(min=1e-8))

        gt_labels = past_labels
        loss = self.loss(log_softmax, gt_labels.long())
        return loss

class SemanticLoss(nn.Module):
    def __init__(self,n_semantic_class,seg_num_per_class,ignore_semantic_index):
        super().__init__()
        self.n_semantic_class = n_semantic_class
        self.ignore_index = ignore_semantic_index
        if seg_num_per_class!='None':
            self.weight = [num_per_class/sum(seg_num_per_class) for num_per_class in seg_num_per_class]
            self.weight = np.power(np.amax(self.weight) / self.weight, 1 / 3.0)

            self.weight=np.insert(self.weight,0,0)# for first ignore
        else:
            self.weight = [0.0 if i in ignore_semantic_index else 1.0 for i in range(n_semantic_class)]
        
        self.weight = torch.Tensor(self.weight)
        self.loss = nn.CrossEntropyLoss(self.weight)


    def compute_loss(self, preb, gt):
        gt = gt.long()
        loss = self.loss(preb,gt)
        return loss

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params:
        num: int,the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.step = 0


    def forward(self, *x):
        loss_sum = 0
        # self.step = self.step+1
        # if self.step%100==0:
        #     self.step=0
        #     print("weight params:",self.params)
        for i, loss in enumerate(x):
            loss_sum =  loss_sum + 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i]**2)
        return loss_sum